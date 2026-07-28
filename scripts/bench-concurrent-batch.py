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
import ctypes
import dataclasses
import datetime as dt
import errno
import hashlib
import io
import json
import math
import os
import platform
import re
import signal
import socket
import stat
import subprocess
import sys
import tempfile
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from contextlib import redirect_stderr
from pathlib import Path
from typing import Any, Callable, Iterable


SCHEMA = "kiln.serving-benchmark.v1"
WORKLOAD_SCHEMA = "kiln.serving-benchmark-workload.v1"
SERVER_LAUNCH_SCHEMA = "kiln.serving-benchmark-server-launch.v1"
DRIVER_VERSION = "27"
PREVIOUS_DRIVER_VERSION = "26"
SUPPORTED_DRIVER_VERSIONS = {PREVIOUS_DRIVER_VERSION, DRIVER_VERSION}
MODERN_DRIVER_VERSIONS = set(SUPPORTED_DRIVER_VERSIONS)
LIFECYCLE_DRIVER_VERSIONS = set(SUPPORTED_DRIVER_VERSIONS)
OUTPUT_EVIDENCE_DRIVER_VERSIONS = set(SUPPORTED_DRIVER_VERSIONS)
ROUTE_AWARE_DIAGNOSTICS_DRIVER_VERSIONS = set(SUPPORTED_DRIVER_VERSIONS)
ROCM_GRAPH_DIAGNOSTICS_DRIVER_VERSIONS = set(SUPPORTED_DRIVER_VERSIONS)
REFERENCE_COMPATIBLE_DRIVER_VERSIONS = set(SUPPORTED_DRIVER_VERSIONS)
COOPERATIVE_ACTOR_CYCLE_IDLE_DRIVER_VERSIONS = set(SUPPORTED_DRIVER_VERSIONS)
MULTI_ROW_GRAPH_FALLBACK_DRIVER_VERSIONS = set(SUPPORTED_DRIVER_VERSIONS)
REQUEST_PERFORMANCE_DRIVER_VERSIONS = set(SUPPORTED_DRIVER_VERSIONS)
PROMPT_SET_IDENTITY_DRIVER_VERSIONS = set(SUPPORTED_DRIVER_VERSIONS)
GRAPH_PARITY_DRIVER_VERSIONS = set(SUPPORTED_DRIVER_VERSIONS)
REFERENCE_ROLE_DRIVER_VERSIONS = set(SUPPORTED_DRIVER_VERSIONS)
ACTOR_ONLY_DIAGNOSTICS_DRIVER_VERSIONS = set(SUPPORTED_DRIVER_VERSIONS)
TYPED_MEMORY_SOURCE_DRIVER_VERSIONS = set(SUPPORTED_DRIVER_VERSIONS)
MACOS_MEMORY_SOURCE_DRIVER_VERSIONS = {DRIVER_VERSION}
REFERENCE_ROLES = {
    "qualification_gate",
    "same_artifact_graph_eager_discriminator",
}
OUTPUT_EVIDENCE_MAX_UTF8_BYTES_PER_REQUEST = 1024 * 1024
LEGACY_PROMPT_TEMPLATE_VERSION = "equal-token-multiset-v1"
FIXED_PROMPT_TEMPLATE_VERSION_V1 = "fixed-serving-profiles-v1"
PROMPT_TEMPLATE_VERSION = "fixed-serving-profiles-v2"
FIXED_PROMPT_TEMPLATE_V2_DRIVER_VERSIONS = set(SUPPORTED_DRIVER_VERSIONS)
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
QUALIFICATION_DIR = ROOT / "scripts" / "qualification"
if str(QUALIFICATION_DIR) not in sys.path:
    sys.path.insert(0, str(QUALIFICATION_DIR))

from scripts import vllm_teacher as teacher  # noqa: E402
from device_memory_sampler import (  # noqa: E402
    DeviceMemoryError,
    MemorySampler,
    resolve_memory_counter,
)
from model_fingerprint import (  # noqa: E402
    ModelFingerprintError,
    fingerprint_model,
)
from request_latency_contract import (  # noqa: E402
    LATENCY_PHASE_FIELDS as REQUEST_PHASE_FIELDS,
    LATENCY_STALL_REASON_FIELDS as REQUEST_STALL_REASON_FIELDS,
)
from strict_json import loads as strict_json_loads  # noqa: E402

DEFAULT_MODEL_FINGERPRINT_READ_MIB_PER_SECOND = 0
MIN_MODEL_FINGERPRINT_READ_MIB_PER_SECOND = 64
MAX_MODEL_FINGERPRINT_READ_MIB_PER_SECOND = 16_384

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
LONG_PROMPT_REPETITIONS_V1 = 64
# Qwen3.5 tokenizes each block to 61 tokens. Sixty-one repetitions keep the
# longest fixed c1 prompt plus 64 output tokens inside the 62x64-token KV pool.
LONG_PROMPT_REPETITIONS = 61

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

REQUEST_COUNTER_FIELDS = ("total", "ok", "error", "timeout", "rejected")
DECODE_BATCHER_COUNTER_FIELDS = (
    "submitted_jobs",
    "executed_batches",
    "executed_rows",
    "runner_calls",
    "runner_busy_jobs",
    "failed_jobs",
)
DIRECT_RENDEZVOUS_FIELDS = (
    "scope",
    "backend_available",
    "backend_unavailable_reason",
    "actor_active",
    "worker_active",
    "route_available",
)
SERVER_DIAGNOSTICS_SCHEMA_V2 = "kiln.serving-benchmark-server-diagnostics.v2"
SERVER_DIAGNOSTICS_SCHEMA_V3 = "kiln.serving-benchmark-server-diagnostics.v3"
SERVER_DIAGNOSTICS_SCHEMA_V4 = "kiln.serving-benchmark-server-diagnostics.v4"
SERVER_DIAGNOSTICS_SCHEMA_V5 = "kiln.serving-benchmark-server-diagnostics.v5"
SERVER_DIAGNOSTICS_SCHEMA_V6 = "kiln.serving-benchmark-server-diagnostics.v6"
SERVER_DIAGNOSTICS_SCHEMA = "kiln.serving-benchmark-server-diagnostics.v7"
SERVER_DIAGNOSTICS_SCHEMAS = {
    SERVER_DIAGNOSTICS_SCHEMA_V2,
    SERVER_DIAGNOSTICS_SCHEMA_V3,
    SERVER_DIAGNOSTICS_SCHEMA_V4,
    SERVER_DIAGNOSTICS_SCHEMA_V5,
    SERVER_DIAGNOSTICS_SCHEMA_V6,
    SERVER_DIAGNOSTICS_SCHEMA,
}
ROCM_GRAPH_COUNTER_FIELDS = (
    "capture_attempts",
    "capture_successes",
    "capture_deferrals",
    "capture_failures",
    "replay_attempts",
    "replay_successes",
    "replay_failures",
    "failures",
    "graph_slot_create_count",
    "graph_slot_reuse_count",
    "cache_admission_successes",
)
ROCM_GRAPH_BATCHED_CAPTURE_COUNTER_FIELDS = (
    "batched_capture_attempts",
    "batched_capture_successes",
    "batched_capture_deferrals",
    "batched_capture_failures",
)
ROCM_GRAPH_CAPTURE_PARITY_COUNTER_FIELDS = (
    "capture_parity_checks",
    "capture_parity_passes",
    "capture_parity_failures",
    "capture_parity_errors",
    "capture_parity_compared_bytes",
    "capture_parity_duration_micros",
)
ROCM_GRAPH_CAPTURE_PARITY_BOUNDARY_FIELDS = (
    "batched_capture_successes",
    "capture_parity_checks",
    "capture_parity_passes",
    "capture_parity_failures",
    "capture_parity_errors",
)
ROCM_GRAPH_GAUGE_FIELDS = (
    "captured_graph_count",
    "graph_slot_count",
    "active_graph_slot_count",
    "idle_graph_slot_count",
)
ROCM_GRAPH_FALLBACK_REASON_FIELDS_V4 = (
    "cold_cache_host_round_trip",
    "persistent_host_round_trip",
    "shape_dependent_attention",
    "graph_cache_capacity",
    "graph_cache_byte_budget",
    "graph_accounting_incomplete",
    "moderate_memory_pressure",
    "tight_memory_pressure",
    "critical_memory_pressure",
    "memory_reservation_denied",
    "memory_governor_selector_mismatch",
    "capture_failure",
    "replay_failure",
)
ROCM_GRAPH_FALLBACK_REASON_FIELDS = (
    "multi_row_batch_unsupported",
    *ROCM_GRAPH_FALLBACK_REASON_FIELDS_V4,
)
ROCM_GRAPH_FALLBACK_COUNTER_FIELDS = (
    "total",
    *ROCM_GRAPH_FALLBACK_REASON_FIELDS,
    "slow",
    "total_duration_micros",
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
    "server_shutdown",
)
COMPLETION_CHECK_STATUSES = {"passed", "failed", "not_applicable"}
COMPLETION_FAILURE_PHASES = {
    "server_startup",
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
OUTPUT_EVIDENCE_KEYS = {
    "index",
    "output_sha256",
    "reasoning_sha256",
    "content_sha256",
    "reasoning_utf8_bytes",
    "content_utf8_bytes",
    "completion_tokens",
    "finish_reason",
    "exact_output",
}
EXACT_OUTPUT_KEYS = {"reasoning_content_base64", "content_base64"}
REQUEST_PERFORMANCE_EVIDENCE_KEYS = {"index", "performance"}
CHAT_PERFORMANCE_KEYS = {
    "prompt_tokens",
    "completion_tokens",
    "ttft_ms",
    "prefill_ms",
    "actor_queue_ms",
    "actor_admission_ms",
    "actor_prefill_wall_ms",
    "resident_prefill_used",
    "decode_ms",
    "total_latency_ms",
    "decode_tokens_per_sec",
    "adapter_used",
    "thinking_mode",
    "finish_reason",
    "latency",
}
REQUEST_LATENCY_KEYS = {
    "emitted_tokens",
    "gap_samples",
    "retained_gap_samples",
    "gap_samples_truncated",
    "ttft_ms",
    "itl_ms_p50",
    "itl_ms_p99",
    "itl_ms_p999",
    "max_itl_ms",
    "stall_threshold_ms",
    "stall_count",
    "unexplained_stall_count",
    "stall_reasons",
    "phases",
}
REQUEST_PERFORMANCE_METRIC_FIELDS = (
    "ttft_ms",
    "prefill_ms",
    "actor_queue_ms",
    "actor_admission_ms",
    "actor_prefill_wall_ms",
    "decode_ms",
    "total_latency_ms",
    "decode_tokens_per_sec",
)
REQUEST_PHASE_SUMMARY_SCHEMA = "kiln.serving-benchmark-request-phase-summary.v1"
REQUEST_PHASE_SUMMARY_KEYS = {
    "schema",
    "performance_request_count",
    "latency_request_count",
    "emitted_tokens",
    "stall_count",
    "unexplained_stall_count",
    "stall_reasons",
    "phases",
    "request_metrics",
}
REQUEST_DISTRIBUTION_KEYS = {
    "observed_request_count",
    "p50",
    "p99",
    "max",
}
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


class OwnedServerShutdownError(BenchmarkError):
    """A shutdown accounting failure with conservative lifecycle evidence."""

    def __init__(self, detail: str, shutdown: dict[str, Any]) -> None:
        super().__init__(detail)
        self.shutdown = shutdown


class OwnedServerLogError(BenchmarkError):
    """A log durability failure with the readable artifact identity retained."""

    def __init__(self, detail: str, log: dict[str, Any]) -> None:
        super().__init__(detail)
        self.log = log


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
    platform_name: str = dataclasses.field(default=sys.platform, repr=False)

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
        expected_command: tuple[str, ...] | None = None,
    ) -> "AttachedProcessGroup":
        if isinstance(pid, bool) or not isinstance(pid, int) or pid <= 1:
            raise BenchmarkError("server PID must be an integer greater than one")
        if sys.platform == "darwin":
            if expected_command is None:
                raise BenchmarkError(
                    "attaching a macOS server requires its exact owned launch command"
                )
            process_group_id, start_time_ticks, executable = (
                _darwin_process_identity(pid)
            )
            if process_group_id != pid:
                raise BenchmarkError(
                    f"server PID {pid} must lead its process group; observed PGID "
                    f"{process_group_id}"
                )
            expected_executable = Path(expected_command[0]).resolve()
            if Path(executable).resolve() != expected_executable:
                raise BenchmarkError(
                    "owned macOS server executable disagrees with its launch command"
                )
            boot = subprocess.run(
                ["/usr/sbin/sysctl", "-n", "kern.boottime"],
                stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                check=False,
                timeout=5.0,
            )
            if boot.returncode != 0 or not boot.stdout.strip():
                raise BenchmarkError("cannot bind the macOS boot identity")
            command_bytes = b"\0".join(
                item.encode("utf-8") for item in expected_command
            ) + b"\0"
            return cls(
                pid=pid,
                process_group_id=process_group_id,
                start_time_ticks=start_time_ticks,
                boot_id=boot.stdout.strip(),
                executable=executable,
                cmdline_sha256="sha256:"
                + hashlib.sha256(command_bytes).hexdigest(),
                proc_root=proc_root,
                platform_name=sys.platform,
            )
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
            platform_name=sys.platform,
        )

    def poll(self) -> int | None:
        if self.platform_name == "darwin":
            try:
                process_group_id, start_time_ticks, executable = (
                    _darwin_process_identity(self.pid)
                )
            except BenchmarkError:
                return 0
            if (
                process_group_id != self.process_group_id
                or start_time_ticks != self.start_time_ticks
                or Path(executable).resolve() != Path(self.executable).resolve()
            ):
                return 0
            return None
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


class _DarwinProcBsdInfo(ctypes.Structure):
    _fields_ = [
        ("pbi_flags", ctypes.c_uint32),
        ("pbi_status", ctypes.c_uint32),
        ("pbi_xstatus", ctypes.c_uint32),
        ("pbi_pid", ctypes.c_uint32),
        ("pbi_ppid", ctypes.c_uint32),
        ("pbi_uid", ctypes.c_uint32),
        ("pbi_gid", ctypes.c_uint32),
        ("pbi_ruid", ctypes.c_uint32),
        ("pbi_rgid", ctypes.c_uint32),
        ("pbi_svuid", ctypes.c_uint32),
        ("pbi_svgid", ctypes.c_uint32),
        ("rfu_1", ctypes.c_uint32),
        ("pbi_comm", ctypes.c_char * 16),
        ("pbi_name", ctypes.c_char * 32),
        ("pbi_nfiles", ctypes.c_uint32),
        ("pbi_pgid", ctypes.c_uint32),
        ("pbi_pjobc", ctypes.c_uint32),
        ("e_tdev", ctypes.c_uint32),
        ("e_tpgid", ctypes.c_uint32),
        ("pbi_nice", ctypes.c_int32),
        ("pbi_start_tvsec", ctypes.c_uint64),
        ("pbi_start_tvusec", ctypes.c_uint64),
    ]


def _darwin_process_identity(pid: int) -> tuple[int, int, str]:
    try:
        library = ctypes.CDLL("/usr/lib/libproc.dylib")
    except OSError as exc:
        raise BenchmarkError(f"cannot load macOS libproc: {exc}") from exc
    library.proc_pidinfo.argtypes = [
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_uint64,
        ctypes.c_void_p,
        ctypes.c_int,
    ]
    library.proc_pidinfo.restype = ctypes.c_int
    info = _DarwinProcBsdInfo()
    size = library.proc_pidinfo(
        pid,
        3,
        0,
        ctypes.byref(info),
        ctypes.sizeof(info),
    )
    if size != ctypes.sizeof(info) or info.pbi_pid != pid:
        raise BenchmarkError(f"cannot read macOS process identity for PID {pid}")
    library.proc_pidpath.argtypes = [
        ctypes.c_int,
        ctypes.c_void_p,
        ctypes.c_uint32,
    ]
    library.proc_pidpath.restype = ctypes.c_int
    path_buffer = ctypes.create_string_buffer(4096)
    path_size = library.proc_pidpath(pid, path_buffer, len(path_buffer))
    if path_size <= 0:
        raise BenchmarkError(f"cannot read macOS executable path for PID {pid}")
    executable = path_buffer.value.decode("utf-8", errors="strict")
    start_microseconds = (
        int(info.pbi_start_tvsec) * 1_000_000 + int(info.pbi_start_tvusec)
    )
    if info.pbi_pgid <= 1 or start_microseconds <= 0 or not executable:
        raise BenchmarkError(f"macOS process identity for PID {pid} is incomplete")
    return int(info.pbi_pgid), start_microseconds, executable


SERVER_LAUNCH_KEYS = {
    "schema",
    "id",
    "command",
    "working_directory",
    "log_directory",
    "readiness_poll_interval_ms",
    "startup_timeout_seconds",
    "shutdown_timeout_seconds",
    "acceptable_exit_codes",
}

@dataclasses.dataclass(frozen=True)
class ServerLaunchConfig:
    record: dict[str, Any]
    command: tuple[str, ...]
    working_directory: Path
    log_directory: Path
    readiness_poll_interval_seconds: float
    startup_timeout_seconds: float
    shutdown_timeout_seconds: float
    acceptable_exit_codes: tuple[int, ...]


def validate_server_launch_config_value(
    value: Any,
    *,
    config_directory: Path,
    label: str,
    require_local_paths: bool = True,
) -> ServerLaunchConfig:
    value = _object(value, label)
    has_content_hash = "content_sha256" in value
    _exact_keys(
        value,
        SERVER_LAUNCH_KEYS | ({"content_sha256"} if has_content_hash else set()),
        label,
    )
    raw = dict(value)
    recorded_hash = raw.pop("content_sha256", None)
    if raw["schema"] != SERVER_LAUNCH_SCHEMA:
        raise BenchmarkError(
            f"serving benchmark launch config must use schema {SERVER_LAUNCH_SCHEMA}"
        )
    if not isinstance(raw["id"], str) or re.fullmatch(
        r"[a-z0-9][a-z0-9._-]{2,127}", raw["id"]
    ) is None:
        raise BenchmarkError("server launch config id must be a portable identifier")
    command = raw["command"]
    if (
        not isinstance(command, list)
        or not command
        or len(command) > 256
        or any(not isinstance(item, str) or not item or "\x00" in item for item in command)
    ):
        raise BenchmarkError(
            "server launch config command must be 1..256 non-empty argv strings"
        )
    for name in ("working_directory", "log_directory"):
        if not isinstance(raw[name], str) or not raw[name] or "\x00" in raw[name]:
            raise BenchmarkError(f"server launch config {name} must be a non-empty path")
    working_directory = Path(raw["working_directory"])
    if not working_directory.is_absolute():
        working_directory = config_directory / working_directory
    working_directory = working_directory.resolve()
    if require_local_paths and not working_directory.is_dir():
        raise BenchmarkError(
            f"server launch working directory is not a directory: {working_directory}"
        )
    executable = Path(command[0])
    if not executable.is_absolute():
        if "/" not in command[0]:
            raise BenchmarkError(
                "server launch executable must be an absolute or explicitly relative path"
            )
        executable = working_directory / executable
    executable = executable.resolve()
    if require_local_paths and (
        not executable.is_file() or not os.access(executable, os.X_OK)
    ):
        raise BenchmarkError(
            f"server launch executable is not a regular executable file: {executable}"
        )
    log_directory = Path(raw["log_directory"])
    if not log_directory.is_absolute():
        log_directory = config_directory / log_directory
    log_directory = log_directory.resolve()
    if require_local_paths and log_directory.exists() and not log_directory.is_dir():
        raise BenchmarkError(
            f"server launch log directory is not a directory: {log_directory}"
        )
    poll_ms = _positive_int(
        raw["readiness_poll_interval_ms"],
        f"{label}.readiness_poll_interval_ms",
    )
    if poll_ms > 60_000:
        raise BenchmarkError("server readiness poll interval must not exceed 60000 ms")
    startup_timeout = _nonnegative_number(
        raw["startup_timeout_seconds"], f"{label}.startup_timeout_seconds"
    )
    shutdown_timeout = _nonnegative_number(
        raw["shutdown_timeout_seconds"], f"{label}.shutdown_timeout_seconds"
    )
    if startup_timeout <= 0 or shutdown_timeout <= 0:
        raise BenchmarkError("server startup and shutdown timeouts must be positive")
    acceptable = raw["acceptable_exit_codes"]
    if (
        not isinstance(acceptable, list)
        or not acceptable
        or len(acceptable) > 16
        or any(isinstance(code, bool) or not isinstance(code, int) for code in acceptable)
        or acceptable != sorted(set(acceptable))
    ):
        raise BenchmarkError(
            "server acceptable exit codes must be a non-empty sorted unique integer array"
        )
    content_hash = canonical_sha256(raw)
    if recorded_hash is not None and _sha256(
        recorded_hash, f"{label}.content_sha256"
    ) != content_hash:
        raise BenchmarkError(f"{label}.content_sha256 does not match launch content")
    normalized = dict(raw)
    normalized["content_sha256"] = content_hash
    return ServerLaunchConfig(
        record=normalized,
        command=(str(executable), *command[1:]),
        working_directory=working_directory,
        log_directory=log_directory,
        readiness_poll_interval_seconds=poll_ms / 1000.0,
        startup_timeout_seconds=startup_timeout,
        shutdown_timeout_seconds=shutdown_timeout,
        acceptable_exit_codes=tuple(acceptable),
    )


def load_server_launch_config(path: Path) -> ServerLaunchConfig:
    if path.is_symlink() or not path.is_file():
        raise BenchmarkError(f"server launch config is not a regular file: {path}")
    data = path.read_bytes()
    if len(data) > 64 * 1024:
        raise BenchmarkError("server launch config exceeds 64 KiB")
    try:
        raw = strict_json_loads(data)
    except Exception as exc:
        raise BenchmarkError(f"cannot load server launch config {path}: {exc}") from exc
    return validate_server_launch_config_value(
        raw,
        config_directory=path.resolve().parent,
        label="server launch config",
    )


def validate_vllm_owned_launch(
    config: ServerLaunchConfig,
    runtime_manifest: dict[str, Any] | None = None,
) -> argparse.Namespace:
    """Bind an owned vLLM launch to the tracked immutable teacher boundary."""

    command = list(config.command)
    if len(command) < 3:
        raise BenchmarkError("owned vLLM launch command is incomplete")
    script_path = Path(command[1])
    candidate = (
        script_path
        if script_path.is_absolute()
        else config.working_directory / script_path
    )
    if candidate.is_symlink() or not candidate.is_file():
        raise BenchmarkError(
            f"owned vLLM launch teacher is not a regular non-symlink file: {candidate}"
        )
    if candidate.resolve() != (ROOT / "scripts" / "vllm_teacher.py").resolve():
        raise BenchmarkError(
            "owned vLLM launch must execute the tracked scripts/vllm_teacher.py"
        )
    parser_error = io.StringIO()
    try:
        with redirect_stderr(parser_error):
            args = teacher.validate_owned_launch_args(command[2:])
    except (SystemExit, teacher.TeacherLaunchError) as exc:
        details = parser_error.getvalue().strip()
        suffix = f": {details}" if details else f": {exc}"
        raise BenchmarkError("invalid owned vLLM launch arguments" + suffix) from exc
    if runtime_manifest is not None:
        identity = runtime_manifest["identity"]
        expected = {
            "served_model_id": args.served_model_id,
            "max_top_k": args.max_top_k,
            "max_model_len": args.max_model_len,
        }
        for field, value in expected.items():
            if identity.get(field) != value:
                raise BenchmarkError(
                    f"owned vLLM launch {field} disagrees with runtime manifest"
                )
    return args


@dataclasses.dataclass
class OwnedServer:
    process: subprocess.Popen[bytes]
    identity: AttachedProcessGroup
    config: ServerLaunchConfig
    log_path: Path
    log_handle: Any


def owned_server_log_path(log_directory: Path, run_id: str) -> Path:
    return log_directory / f"{run_id}.server.log"


def launch_owned_server(config: ServerLaunchConfig, run_id: str) -> OwnedServer:
    log_path = owned_server_log_path(config.log_directory, run_id)
    config.log_directory.mkdir(parents=True, exist_ok=True)
    try:
        log_handle = log_path.open("xb", buffering=0)
    except OSError as exc:
        raise BenchmarkError(f"cannot create server log {log_path}: {exc}") from exc
    try:
        process = subprocess.Popen(
            list(config.command),
            cwd=config.working_directory,
            stdin=subprocess.DEVNULL,
            stdout=log_handle,
            stderr=subprocess.STDOUT,
            start_new_session=True,
            close_fds=True,
        )
        try:
            attach_deadline = time.monotonic() + 2.0
            while True:
                if process.poll() is not None:
                    raise BenchmarkError(
                        f"owned server exited with status {process.returncode} "
                        "before process identity could be bound"
                    )
                try:
                    identity = (
                        AttachedProcessGroup.attach(
                            process.pid,
                            expected_command=config.command,
                        )
                        if sys.platform == "darwin"
                        else AttachedProcessGroup.attach(process.pid)
                    )
                    break
                except BenchmarkError:
                    if time.monotonic() >= attach_deadline:
                        raise
                    time.sleep(0.005)
        except Exception:
            try:
                os.killpg(process.pid, signal.SIGTERM)
            except OSError:
                pass
            try:
                process.wait(timeout=5.0)
            except subprocess.TimeoutExpired:
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except OSError:
                    pass
                process.wait(timeout=5.0)
            raise
        return OwnedServer(
            process=process,
            identity=identity,
            config=config,
            log_path=log_path,
            log_handle=log_handle,
        )
    except Exception:
        log_handle.close()
        raise


def _process_group_member_states(
    process_group_id: int,
    proc_root: Path,
) -> tuple[list[str], bool]:
    states: list[str] = []
    certain = True
    try:
        process_paths = list(proc_root.iterdir())
    except OSError:
        return states, False
    for process_path in process_paths:
        if not process_path.name.isdigit():
            continue
        try:
            raw = (process_path / "stat").read_text(encoding="utf-8")
        except (FileNotFoundError, ProcessLookupError):
            continue
        except OSError:
            certain = False
            continue
        close = raw.rfind(")")
        if close < 0:
            certain = False
            continue
        fields = raw[close + 2 :].split()
        if len(fields) < 3:
            certain = False
            continue
        try:
            observed_group = int(fields[2])
        except ValueError:
            certain = False
            continue
        if observed_group == process_group_id:
            states.append(fields[0])
    return states, certain


def process_group_alive(
    process_group_id: int,
    proc_root: Path = Path("/proc"),
) -> bool:
    # killpg(..., 0) also succeeds for a zombie-only group. Such members cannot
    # execute or retain descriptors, but any uncertain or non-zombie member
    # still means cleanup is incomplete.
    try:
        os.killpg(process_group_id, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    if sys.platform == "darwin":
        return True
    states, certain = _process_group_member_states(process_group_id, proc_root)
    if not certain or not states:
        return True
    return any(state != "Z" for state in states)


def loopback_base_url_port(base_url: str) -> int:
    parsed = urllib.parse.urlsplit(base_url)
    if (
        parsed.scheme != "http"
        or parsed.hostname not in {"127.0.0.1", "localhost", "::1"}
        or parsed.username is not None
        or parsed.password is not None
        or parsed.path not in {"", "/"}
        or parsed.query
        or parsed.fragment
    ):
        raise BenchmarkError(
            "owned server base URL must be an origin-only loopback HTTP URL"
        )
    try:
        return parsed.port or 80
    except ValueError as exc:
        raise BenchmarkError(f"owned server base URL has an invalid port: {exc}") from exc


def listening_socket_inodes(port: int, proc_root: Path = Path("/proc")) -> set[str]:
    inodes: set[str] = set()
    for name in ("tcp", "tcp6"):
        path = proc_root / "net" / name
        try:
            lines = path.read_text(encoding="utf-8").splitlines()[1:]
        except OSError as exc:
            raise BenchmarkError(f"cannot inspect listening sockets in {path}: {exc}") from exc
        for line in lines:
            fields = line.split()
            if len(fields) < 10 or fields[3] != "0A":
                continue
            try:
                local_port = int(fields[1].rsplit(":", 1)[1], 16)
            except (IndexError, ValueError) as exc:
                raise BenchmarkError(f"malformed socket row in {path}: {line}") from exc
            if local_port == port:
                inodes.add(fields[9])
    return inodes


def process_group_socket_inodes(
    process_group_id: int, proc_root: Path = Path("/proc")
) -> set[str]:
    inodes: set[str] = set()
    try:
        process_paths = list(proc_root.iterdir())
    except OSError as exc:
        raise BenchmarkError(f"cannot enumerate {proc_root}: {exc}") from exc
    for process_path in process_paths:
        if not process_path.name.isdigit():
            continue
        try:
            _state, observed_group, _start = AttachedProcessGroup._read_stat(
                int(process_path.name), proc_root
            )
        except BenchmarkError:
            continue
        if observed_group != process_group_id:
            continue
        try:
            descriptors = list((process_path / "fd").iterdir())
        except OSError:
            continue
        for descriptor in descriptors:
            try:
                target = os.readlink(descriptor)
            except OSError:
                continue
            match = re.fullmatch(r"socket:\[(\d+)\]", target)
            if match is not None:
                inodes.add(match.group(1))
    return inodes


def require_owned_base_url_unbound(base_url: str) -> int:
    port = loopback_base_url_port(base_url)
    if sys.platform == "darwin":
        parsed = urllib.parse.urlsplit(base_url)
        family = socket.AF_INET6 if parsed.hostname == "::1" else socket.AF_INET
        address = (
            ("::1", port, 0, 0)
            if family == socket.AF_INET6
            else ("127.0.0.1", port)
        )
        try:
            with socket.socket(family, socket.SOCK_STREAM) as probe:
                probe.bind(address)
        except OSError as exc:
            raise BenchmarkError(
                f"owned server base URL port {port} is already listening or unavailable: {exc}"
            ) from exc
        return port
    if listening_socket_inodes(port):
        raise BenchmarkError(
            f"owned server base URL port {port} is already listening before launch"
        )
    return port


def verify_owned_listener(server: OwnedServer, base_url: str) -> None:
    port = loopback_base_url_port(base_url)
    if sys.platform == "darwin":
        if server.process.poll() is not None:
            raise BenchmarkError(
                "owned macOS server exited before listener verification"
            )
        parsed = urllib.parse.urlsplit(base_url)
        host = "::1" if parsed.hostname == "::1" else "127.0.0.1"
        try:
            with socket.create_connection((host, port), timeout=2.0):
                pass
        except OSError as exc:
            raise BenchmarkError(
                f"owned macOS server port {port} is not reachable after readiness: {exc}"
            ) from exc
        return
    listeners = listening_socket_inodes(port)
    owned = process_group_socket_inodes(server.identity.process_group_id)
    if not listeners:
        raise BenchmarkError(
            f"owned server base URL port {port} has no listening socket after readiness"
        )
    if listeners.isdisjoint(owned):
        raise BenchmarkError(
            f"owned server base URL port {port} is not owned by process group "
            f"{server.identity.process_group_id}"
        )


def _wait_for_owned_process(
    process: subprocess.Popen[bytes],
    timeout_seconds: float,
) -> int | None:
    wall_deadline = time.monotonic() + timeout_seconds
    while True:
        returncode = process.poll()
        if returncode is not None:
            return process.wait()
        now = time.monotonic()
        if now >= wall_deadline:
            return None
        time.sleep(0.05)


def _wait_for_process_group_exit(
    process_group_id: int,
    timeout_seconds: float,
) -> bool:
    wall_deadline = time.monotonic() + timeout_seconds
    while process_group_alive(process_group_id):
        now = time.monotonic()
        if now >= wall_deadline:
            return False
        time.sleep(0.05)
    return True


def _emergency_force_drain_owned_server(
    server: OwnedServer,
) -> tuple[int | None, bool]:
    """Best-effort wall-bounded cleanup after trusted timing becomes unavailable."""

    try:
        os.killpg(server.identity.process_group_id, signal.SIGKILL)
    except (ProcessLookupError, PermissionError):
        pass
    try:
        returncode = server.process.wait(timeout=10.0)
    except subprocess.TimeoutExpired:
        returncode = server.process.poll()
    group_exited = _wait_for_process_group_exit(
        server.identity.process_group_id,
        10.0,
    )
    return returncode, group_exited


def shutdown_owned_server(
    server: OwnedServer,
) -> dict[str, Any]:
    started = time.monotonic()
    signal_sent = server.process.poll() is None
    try:
        return _shutdown_owned_server(server)
    except BaseException as exc:
        returncode, group_exited = _emergency_force_drain_owned_server(server)
        if returncode is None or not group_exited:
            raise BenchmarkError(
                "owned server cleanup failed after shutdown accounting error: "
                f"{exc}"
            ) from exc
        if not isinstance(exc, Exception):
            raise
        shutdown = {
            "signal": "SIGTERM",
            "signal_sent": signal_sent,
            "forced": True,
            "returncode": returncode,
            "acceptable_exit_codes": list(server.config.acceptable_exit_codes),
            "elapsed_seconds": time.monotonic() - started,
            "process_group_alive_end": False,
        }
        raise OwnedServerShutdownError(
            "owned server shutdown accounting failed after bounded force drain: "
            f"{type(exc).__name__}: {exc}",
            shutdown,
        ) from exc


def _shutdown_owned_server(
    server: OwnedServer,
) -> dict[str, Any]:
    started = time.monotonic()
    signal_sent = server.process.poll() is None
    forced = False
    if signal_sent:
        try:
            os.killpg(server.process.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
    returncode = _wait_for_owned_process(
        server.process,
        server.config.shutdown_timeout_seconds,
    )
    if returncode is None:
        forced = True
        try:
            os.killpg(server.process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        returncode = _wait_for_owned_process(
            server.process,
            10.0,
        )
        if returncode is None:
            raise BenchmarkError(
                "owned server process did not exit after SIGTERM and SIGKILL "
                "within 10.000 wall seconds"
            )
    group_exited = _wait_for_process_group_exit(
        server.identity.process_group_id,
        10.0,
    )
    if not group_exited:
        forced = True
        try:
            os.killpg(server.identity.process_group_id, signal.SIGKILL)
        except ProcessLookupError:
            pass
        group_exited = _wait_for_process_group_exit(
            server.identity.process_group_id,
            10.0,
        )
        if not group_exited:
            raise BenchmarkError(
                "owned server process group survived SIGTERM and SIGKILL"
            )
    return {
        "signal": "SIGTERM",
        "signal_sent": signal_sent,
        "forced": forced,
        "returncode": returncode,
        "acceptable_exit_codes": list(server.config.acceptable_exit_codes),
        "elapsed_seconds": time.monotonic() - started,
        "process_group_alive_end": not group_exited,
    }


TRANSIENT_LOG_FSYNC_ERRNOS = {
    errno.EAGAIN,
    errno.EBUSY,
    errno.EINTR,
    errno.ETIMEDOUT,
}
LOG_FSYNC_ATTEMPTS = 3


def _fsync_owned_server_log(handle: Any) -> None:
    for attempt in range(1, LOG_FSYNC_ATTEMPTS + 1):
        try:
            os.fsync(handle.fileno())
            return
        except OSError as exc:
            if (
                exc.errno not in TRANSIENT_LOG_FSYNC_ERRNOS
                or attempt == LOG_FSYNC_ATTEMPTS
            ):
                raise
            time.sleep(0.05)


def _owned_server_log_identity(path: Path) -> dict[str, Any]:
    path = path.absolute()
    before = path.stat(follow_symlinks=False)
    if not stat.S_ISREG(before.st_mode):
        raise BenchmarkError(f"server log is not a regular file: {path}")
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    after = path.stat(follow_symlinks=False)
    if _stat_identity(before) != _stat_identity(after):
        raise BenchmarkError(f"server log changed while hashing: {path}")
    return {
        "path": str(path),
        "bytes": before.st_size,
        "sha256": "sha256:" + digest.hexdigest(),
    }


def close_owned_server_log(server: OwnedServer) -> dict[str, Any]:
    durability_error: Exception | None = None
    if not server.log_handle.closed:
        try:
            server.log_handle.flush()
            _fsync_owned_server_log(server.log_handle)
        except Exception as exc:
            durability_error = exc
        finally:
            try:
                server.log_handle.close()
            except Exception as exc:
                if durability_error is None:
                    durability_error = exc
                else:
                    durability_error = BenchmarkError(
                        f"{type(durability_error).__name__}: {durability_error}; "
                        f"log close also failed: {type(exc).__name__}: {exc}"
                    )
    log = _owned_server_log_identity(server.log_path)
    if durability_error is not None:
        raise OwnedServerLogError(
            "owned server log durability failed before artifact hashing: "
            f"{type(durability_error).__name__}: {durability_error}",
            log,
        ) from durability_error
    return log


def finalize_owned_server(
    server: OwnedServer,
) -> tuple[dict[str, Any] | None, dict[str, Any] | None, list[Exception]]:
    shutdown: dict[str, Any] | None = None
    log: dict[str, Any] | None = None
    failures: list[Exception] = []
    try:
        shutdown = shutdown_owned_server(server)
    except OwnedServerShutdownError as exc:
        shutdown = exc.shutdown
        failures.append(exc)
    except Exception as exc:
        failures.append(exc)
    try:
        log = close_owned_server_log(server)
    except OwnedServerLogError as exc:
        log = exc.log
        failures.append(exc)
    except Exception as exc:
        failures.append(exc)
    return shutdown, log, failures


def server_log_tail(path: Path, limit_bytes: int = 16 * 1024) -> str:
    try:
        with path.open("rb") as handle:
            handle.seek(0, os.SEEK_END)
            size = handle.tell()
            handle.seek(max(0, size - limit_bytes))
            return handle.read(limit_bytes).decode("utf-8", errors="replace")[-limit_bytes:]
    except OSError as exc:
        return f"<cannot read server log: {exc}>"


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


def validate_kiln_execution_identity(
    value: Any,
    artifact: dict[str, Any],
    label: str,
) -> dict[str, Any]:
    identity = _object(value, label)
    if identity.get("executable_sha256") != artifact["sha256"]:
        raise BenchmarkError(
            "receipt Kiln execution identity does not bind the runtime artifact"
        )
    if identity.get("provenance_type") != "kiln.execution-provenance.v1":
        raise BenchmarkError("receipt Kiln execution provenance type is unsupported")
    for name in ("backend", "device", "inference_dtype", "training_policy"):
        if not isinstance(identity.get(name), str) or not identity[name]:
            raise BenchmarkError(f"{label}.{name} must be non-empty")
    for name in (
        "provenance_sha256",
        "executable_sha256",
        "numerical_runtime_sha256",
        "kernel_contract_sha256",
        "effective_server_config_sha256",
        "effective_environment_sha256",
    ):
        _sha256(identity.get(name), f"{label}.{name}")
    return identity


def validate_vllm_runtime_manifest(value: Any, label: str) -> dict[str, Any]:
    manifest = _object(value, label)
    _exact_keys(manifest, VLLM_RUNTIME_MANIFEST_KEYS, label)
    identity = _object(manifest["identity"], f"{label}.identity")
    canonical_json = manifest["canonical_json"]
    if not isinstance(canonical_json, str) or not canonical_json:
        raise BenchmarkError(f"{label}.canonical_json must be non-empty")
    try:
        canonical_identity = strict_json_loads(canonical_json.encode("utf-8"))
    except Exception as exc:
        raise BenchmarkError(f"{label}.canonical_json is not strict JSON: {exc}") from exc
    expected_json = json.dumps(
        canonical_identity,
        ensure_ascii=False,
        allow_nan=False,
        separators=(",", ":"),
    )
    if canonical_json != expected_json or canonical_identity != identity:
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


def _decode_canonical_base64_text(value: Any, label: str) -> str:
    if not isinstance(value, str):
        raise BenchmarkError(f"{label} must be a base64 string")
    try:
        encoded = value.encode("ascii")
        raw = base64.b64decode(encoded, validate=True)
    except (UnicodeEncodeError, binascii.Error) as exc:
        raise BenchmarkError(f"{label} must be canonical base64") from exc
    if base64.b64encode(raw) != encoded:
        raise BenchmarkError(f"{label} must use canonical padded base64")
    try:
        return raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise BenchmarkError(f"{label} must encode UTF-8 text") from exc


def output_set_evidence_row(evidence: dict[str, Any]) -> dict[str, Any]:
    return {
        name: evidence[name]
        for name in (
            "index",
            "output_sha256",
            "reasoning_sha256",
            "content_sha256",
            "reasoning_utf8_bytes",
            "content_utf8_bytes",
            "completion_tokens",
            "finish_reason",
        )
    }


def validate_output_evidence(
    value: Any,
    *,
    label: str,
    concurrency: int,
    success_count: int,
    error_indices: set[int],
    completion_tokens: int,
    output_set_sha256: str,
) -> None:
    if not isinstance(value, list) or len(value) != success_count:
        raise BenchmarkError(
            f"{label} must contain exactly one row per successful request"
        )
    seen: set[int] = set()
    previous_index = -1
    aggregate_rows: list[dict[str, Any]] = []
    exact_modes: set[bool] = set()
    evidence_completion_tokens = 0
    for position, evidence_value in enumerate(value):
        evidence_label = f"{label}[{position}]"
        evidence = _object(evidence_value, evidence_label)
        _exact_keys(evidence, OUTPUT_EVIDENCE_KEYS, evidence_label)
        index = evidence["index"]
        if (
            isinstance(index, bool)
            or not isinstance(index, int)
            or not 0 <= index < concurrency
            or index in seen
            or index in error_indices
        ):
            raise BenchmarkError(f"{evidence_label}.index is invalid or duplicate")
        if index <= previous_index:
            raise BenchmarkError(f"{label} must be ordered by request index")
        previous_index = index
        seen.add(index)
        for name in ("output_sha256", "reasoning_sha256", "content_sha256"):
            _sha256(evidence[name], f"{evidence_label}.{name}")
        reasoning_bytes = _nonnegative_int(
            evidence["reasoning_utf8_bytes"],
            f"{evidence_label}.reasoning_utf8_bytes",
        )
        content_bytes = _nonnegative_int(
            evidence["content_utf8_bytes"],
            f"{evidence_label}.content_utf8_bytes",
        )
        request_tokens = _positive_int(
            evidence["completion_tokens"], f"{evidence_label}.completion_tokens"
        )
        evidence_completion_tokens += request_tokens
        if (
            not isinstance(evidence["finish_reason"], str)
            or not evidence["finish_reason"]
        ):
            raise BenchmarkError(f"{evidence_label}.finish_reason must be non-empty")
        exact = evidence["exact_output"]
        exact_modes.add(exact is not None)
        if exact is not None:
            if (
                reasoning_bytes + content_bytes
                > OUTPUT_EVIDENCE_MAX_UTF8_BYTES_PER_REQUEST
            ):
                raise BenchmarkError(
                    f"{evidence_label} exceeds the per-request evidence bound"
                )
            exact_object = _object(exact, f"{evidence_label}.exact_output")
            _exact_keys(
                exact_object,
                EXACT_OUTPUT_KEYS,
                f"{evidence_label}.exact_output",
            )
            reasoning = _decode_canonical_base64_text(
                exact_object["reasoning_content_base64"],
                f"{evidence_label}.exact_output.reasoning_content_base64",
            )
            content = _decode_canonical_base64_text(
                exact_object["content_base64"],
                f"{evidence_label}.exact_output.content_base64",
            )
            if len(reasoning.encode("utf-8")) != reasoning_bytes:
                raise BenchmarkError(f"{evidence_label} reasoning byte count disagrees")
            if len(content.encode("utf-8")) != content_bytes:
                raise BenchmarkError(f"{evidence_label} content byte count disagrees")
            if text_sha256(reasoning) != evidence["reasoning_sha256"]:
                raise BenchmarkError(f"{evidence_label} reasoning hash disagrees")
            if text_sha256(content) != evidence["content_sha256"]:
                raise BenchmarkError(f"{evidence_label} content hash disagrees")
            if text_sha256(reasoning + "\x1e" + content) != evidence["output_sha256"]:
                raise BenchmarkError(f"{evidence_label} output hash disagrees")
        aggregate_rows.append(output_set_evidence_row(evidence))
    if len(exact_modes) > 1:
        raise BenchmarkError(f"{label} mixes hash-only and full output evidence")
    expected_indices = set(range(concurrency)) - error_indices
    if seen != expected_indices:
        raise BenchmarkError(f"{label} does not cover every successful request")
    if evidence_completion_tokens != completion_tokens:
        raise BenchmarkError(f"{label} completion-token total disagrees with its run")
    if canonical_sha256(aggregate_rows) != output_set_sha256:
        raise BenchmarkError(f"{label} does not reproduce output_set_sha256")


def validate_server_diagnostics_v2(value: Any, label: str) -> dict[str, Any]:
    server = _object(value, label)
    _exact_keys(
        server,
        {
            "schema",
            "request_route",
            "requests",
            "routing",
            "batching_engine",
            "decode_batcher",
        },
        label,
    )
    if server["schema"] != SERVER_DIAGNOSTICS_SCHEMA_V2:
        raise BenchmarkError(f"{label}.schema is unsupported")
    if server["request_route"] not in {"batching_engine", "direct_streaming"}:
        raise BenchmarkError(f"{label}.request_route is unsupported")

    requests = _object(server["requests"], f"{label}.requests")
    request_fields = {
        *REQUEST_COUNTER_FIELDS,
        "active_end",
        "process_active_peak",
    }
    _exact_keys(requests, request_fields, f"{label}.requests")
    for field in request_fields:
        _nonnegative_int(requests[field], f"{label}.requests.{field}")
    if requests["total"] != sum(
        requests[field] for field in REQUEST_COUNTER_FIELDS[1:]
    ):
        raise BenchmarkError(f"{label}.requests status counters disagree")
    if requests["active_end"] > requests["process_active_peak"]:
        raise BenchmarkError(f"{label}.requests active_end exceeds process peak")

    routing = _object(server["routing"], f"{label}.routing")
    _exact_keys(
        routing,
        {"batching_actor_effective", "direct_decode_rendezvous"},
        f"{label}.routing",
    )
    actor_effective = routing["batching_actor_effective"]
    if not isinstance(actor_effective, bool):
        raise BenchmarkError(f"{label}.routing.batching_actor_effective must be boolean")
    direct = _object(
        routing["direct_decode_rendezvous"],
        f"{label}.routing.direct_decode_rendezvous",
    )
    _exact_keys(
        direct,
        set(DIRECT_RENDEZVOUS_FIELDS),
        f"{label}.routing.direct_decode_rendezvous",
    )
    if not isinstance(direct["scope"], str) or not direct["scope"]:
        raise BenchmarkError(
            f"{label}.routing.direct_decode_rendezvous.scope must be non-empty"
        )
    for field in (
        "backend_available",
        "actor_active",
        "worker_active",
        "route_available",
    ):
        if not isinstance(direct[field], bool):
            raise BenchmarkError(
                f"{label}.routing.direct_decode_rendezvous.{field} must be boolean"
            )
    reason = direct["backend_unavailable_reason"]
    if direct["backend_available"]:
        if reason is not None:
            raise BenchmarkError(f"{label} available direct backend has a reason")
    elif not isinstance(reason, str) or not reason:
        raise BenchmarkError(f"{label} unavailable direct backend omits its reason")
    route_available = (
        direct["backend_available"]
        and not direct["actor_active"]
        and direct["worker_active"]
    )
    if direct["route_available"] != route_available:
        raise BenchmarkError(f"{label} direct-route ownership is inconsistent")
    if direct["actor_active"] != actor_effective:
        raise BenchmarkError(f"{label} actor ownership is inconsistent")
    expected_request_route = (
        "batching_engine" if actor_effective else "direct_streaming"
    )
    if server["request_route"] != expected_request_route:
        raise BenchmarkError(f"{label}.request_route disagrees with actor ownership")

    batching = server["batching_engine"]
    if batching is not None:
        batching = _object(batching, f"{label}.batching_engine")
        required_batching = set(COUNTER_FIELDS) | {
            "effective_max_decode_batch",
            "process_max_observed_batch",
            "mean_decode_rows_per_forward",
            "batched_decode_forward_fraction",
        }
        _exact_keys(batching, required_batching, f"{label}.batching_engine")
        for field, item in batching.items():
            _nonnegative_number(item, f"{label}.batching_engine.{field}")
    if (batching is not None) != actor_effective:
        raise BenchmarkError(f"{label}.batching_engine availability is inconsistent")

    decode_batcher = server["decode_batcher"]
    if decode_batcher is not None:
        decode_batcher = _object(decode_batcher, f"{label}.decode_batcher")
        integer_fields = {
            *DECODE_BATCHER_COUNTER_FIELDS,
            "process_max_observed_batch",
            "process_max_runner_calls_per_token",
            "runner_call_budget_per_token",
        }
        required_direct = {
            *integer_fields,
            "mean_runner_calls_per_executed_row",
            "runner_call_budget_exceeded",
        }
        _exact_keys(decode_batcher, required_direct, f"{label}.decode_batcher")
        for field in integer_fields:
            _nonnegative_int(
                decode_batcher[field], f"{label}.decode_batcher.{field}"
            )
        mean_calls = decode_batcher["mean_runner_calls_per_executed_row"]
        if mean_calls is not None:
            _nonnegative_number(
                mean_calls,
                f"{label}.decode_batcher.mean_runner_calls_per_executed_row",
            )
        expected_mean = (
            decode_batcher["runner_calls"] / decode_batcher["executed_rows"]
            if decode_batcher["executed_rows"]
            else None
        )
        if mean_calls != expected_mean:
            raise BenchmarkError(f"{label}.decode_batcher mean runner calls disagree")
        budget_exceeded = decode_batcher["runner_call_budget_exceeded"]
        if not isinstance(budget_exceeded, bool):
            raise BenchmarkError(
                f"{label}.decode_batcher.runner_call_budget_exceeded must be boolean"
            )
        if budget_exceeded != (
            decode_batcher["process_max_runner_calls_per_token"]
            > decode_batcher["runner_call_budget_per_token"]
        ):
            raise BenchmarkError(f"{label}.decode_batcher budget state disagrees")
    if (decode_batcher is not None) != direct["worker_active"]:
        raise BenchmarkError(f"{label}.decode_batcher availability is inconsistent")
    return server


def validate_rocm_graph_record(
    value: Any,
    label: str,
    *,
    gauge_suffixes: tuple[str, ...],
    fallback_max_field: str,
    fallback_reason_fields: tuple[str, ...],
) -> dict[str, Any]:
    graph = _object(value, label)
    configuration_fields = {
        "state",
        "unavailable_reason",
        "requested",
        "capture_requested",
        "enabled",
        "capture_enabled",
    }
    counter_fields = set(ROCM_GRAPH_COUNTER_FIELDS)
    gauge_fields = {
        f"{field}{suffix}"
        for field in ROCM_GRAPH_GAUGE_FIELDS
        for suffix in gauge_suffixes
    }
    _exact_keys(
        graph,
        configuration_fields | counter_fields | gauge_fields | {"fallbacks"},
        label,
    )
    if graph["state"] not in {"enabled", "disabled", "busy", "unavailable"}:
        raise BenchmarkError(f"{label}.state is unsupported")

    available = graph["state"] in {"enabled", "disabled"}
    reason = graph["unavailable_reason"]
    optional_fields = {
        "requested",
        "capture_requested",
        "enabled",
        "capture_enabled",
        *counter_fields,
        *gauge_fields,
    }
    if available:
        if reason is not None:
            raise BenchmarkError(f"{label} available graph diagnostics have a reason")
        for field in (
            "requested",
            "capture_requested",
            "enabled",
            "capture_enabled",
        ):
            if not isinstance(graph[field], bool):
                raise BenchmarkError(f"{label}.{field} must be boolean")
        if graph["enabled"] != (graph["state"] == "enabled"):
            raise BenchmarkError(f"{label}.state disagrees with enabled")
        if graph["capture_enabled"] and not graph["capture_requested"]:
            raise BenchmarkError(
                f"{label}.capture_enabled requires capture_requested"
            )
        for field in counter_fields | gauge_fields:
            _nonnegative_int(graph[field], f"{label}.{field}")
        if graph["failures"] != (
            graph["capture_failures"] + graph["replay_failures"]
        ):
            raise BenchmarkError(f"{label}.failures disagrees with failure classes")
        if graph["replay_attempts"] != (
            graph["replay_successes"] + graph["replay_failures"]
        ):
            raise BenchmarkError(f"{label}.replay attempt accounting disagrees")
        for suffix in gauge_suffixes:
            slots = graph[f"graph_slot_count{suffix}"]
            active = graph[f"active_graph_slot_count{suffix}"]
            idle = graph[f"idle_graph_slot_count{suffix}"]
            if active + idle != slots:
                raise BenchmarkError(
                    f"{label} graph-slot gauges disagree at {suffix or 'snapshot'}"
                )
        fallbacks = _object(graph["fallbacks"], f"{label}.fallbacks")
        fallback_counter_fields = {
            "total",
            *fallback_reason_fields,
            "slow",
            "total_duration_micros",
        }
        _exact_keys(
            fallbacks,
            {*fallback_counter_fields, fallback_max_field},
            f"{label}.fallbacks",
        )
        for field, item in fallbacks.items():
            _nonnegative_int(item, f"{label}.fallbacks.{field}")
        if fallbacks["total"] != sum(
            fallbacks[field] for field in fallback_reason_fields
        ):
            raise BenchmarkError(f"{label}.fallback reason accounting disagrees")
    else:
        if not isinstance(reason, str) or not reason:
            raise BenchmarkError(f"{label} unavailable graph diagnostics omit a reason")
        for field in optional_fields:
            if graph[field] is not None:
                raise BenchmarkError(f"{label}.{field} must be null when unavailable")
        if graph["fallbacks"] is not None:
            raise BenchmarkError(f"{label}.fallbacks must be null when unavailable")
    return graph


def validate_rocm_graph_diagnostics_v4(value: Any, label: str) -> dict[str, Any]:
    return validate_rocm_graph_record(
        value,
        label,
        gauge_suffixes=("_start", "_end"),
        fallback_max_field="process_max_duration_micros",
        fallback_reason_fields=ROCM_GRAPH_FALLBACK_REASON_FIELDS_V4,
    )


def validate_rocm_graph_diagnostics(value: Any, label: str) -> dict[str, Any]:
    return validate_rocm_graph_record(
        value,
        label,
        gauge_suffixes=("_start", "_end"),
        fallback_max_field="process_max_duration_micros",
        fallback_reason_fields=ROCM_GRAPH_FALLBACK_REASON_FIELDS,
    )


def validate_rocm_graph_capture_parity(value: Any, label: str) -> dict[str, Any]:
    parity = _object(value, label)
    delta_fields = {
        *ROCM_GRAPH_BATCHED_CAPTURE_COUNTER_FIELDS,
        *ROCM_GRAPH_CAPTURE_PARITY_COUNTER_FIELDS,
    }
    boundary_fields = {
        f"{field}_{boundary}"
        for field in ROCM_GRAPH_CAPTURE_PARITY_BOUNDARY_FIELDS
        for boundary in ("start", "end")
    }
    _exact_keys(parity, delta_fields | boundary_fields, label)
    for field in delta_fields | boundary_fields:
        _nonnegative_int(parity[field], f"{label}.{field}")
    if parity["batched_capture_attempts"] != (
        parity["batched_capture_successes"]
        + parity["batched_capture_deferrals"]
        + parity["batched_capture_failures"]
    ):
        raise BenchmarkError(f"{label} batched capture outcomes disagree")
    if parity["capture_parity_checks"] != (
        parity["capture_parity_passes"]
        + parity["capture_parity_failures"]
        + parity["capture_parity_errors"]
    ):
        raise BenchmarkError(f"{label} parity outcomes disagree")
    for field in ROCM_GRAPH_CAPTURE_PARITY_BOUNDARY_FIELDS:
        if parity[f"{field}_end"] < parity[f"{field}_start"]:
            raise BenchmarkError(f"{label}.{field} regressed")
        if parity[field] != parity[f"{field}_end"] - parity[f"{field}_start"]:
            raise BenchmarkError(f"{label}.{field} delta disagrees with boundaries")
    for boundary in ("start", "end"):
        checks = parity[f"capture_parity_checks_{boundary}"]
        outcomes = (
            parity[f"capture_parity_passes_{boundary}"]
            + parity[f"capture_parity_failures_{boundary}"]
            + parity[f"capture_parity_errors_{boundary}"]
        )
        if checks != outcomes:
            raise BenchmarkError(
                f"{label} cumulative parity outcomes disagree at {boundary}"
            )
        if (
            parity[f"batched_capture_successes_{boundary}"]
            > parity[f"capture_parity_passes_{boundary}"]
        ):
            raise BenchmarkError(
                f"{label} successful batched captures lack parity at {boundary}"
            )
    if parity["capture_parity_checks"] > 0 and parity["capture_parity_compared_bytes"] == 0:
        raise BenchmarkError(f"{label} parity checks must cover positive bytes")
    return parity


def validate_rocm_graph_diagnostics_v6(value: Any, label: str) -> dict[str, Any]:
    graph = _object(value, label)
    _exact_keys(
        graph,
        {
            "state",
            "unavailable_reason",
            "requested",
            "capture_requested",
            "enabled",
            "capture_enabled",
            *ROCM_GRAPH_COUNTER_FIELDS,
            *{
                f"{field}{suffix}"
                for field in ROCM_GRAPH_GAUGE_FIELDS
                for suffix in ("_start", "_end")
            },
            "fallbacks",
            "capture_parity",
        },
        label,
    )
    legacy = dict(graph)
    parity = legacy.pop("capture_parity")
    validate_rocm_graph_diagnostics(legacy, label)
    if graph["state"] in {"enabled", "disabled"}:
        validate_rocm_graph_capture_parity(parity, f"{label}.capture_parity")
    elif parity is not None:
        raise BenchmarkError(f"{label}.capture_parity must be null when unavailable")
    return graph


def validate_server_diagnostics_v3(value: Any, label: str) -> dict[str, Any]:
    server = _object(value, label)
    _exact_keys(
        server,
        {
            "schema",
            "request_route",
            "requests",
            "routing",
            "batching_engine",
            "decode_batcher",
            "rocm_graphs",
        },
        label,
    )
    if server["schema"] != SERVER_DIAGNOSTICS_SCHEMA_V3:
        raise BenchmarkError(f"{label}.schema is unsupported")
    route_record = {key: value for key, value in server.items() if key != "rocm_graphs"}
    route_record["schema"] = SERVER_DIAGNOSTICS_SCHEMA_V2
    validate_server_diagnostics_v2(route_record, label)
    validate_rocm_graph_diagnostics_v4(server["rocm_graphs"], f"{label}.rocm_graphs")
    return server


def validate_server_diagnostics_v4(value: Any, label: str) -> dict[str, Any]:
    server = _object(value, label)
    if server.get("schema") != SERVER_DIAGNOSTICS_SCHEMA_V4:
        raise BenchmarkError(f"{label}.schema is unsupported")
    batching = server.get("batching_engine")
    if batching is not None:
        batching = _object(batching, f"{label}.batching_engine")
        idle_fields = {
            "actor_cycle_idle_ms",
            "actor_cycle_idle_source",
            "actor_cycle_idle_active_end",
            "actor_cycle_idle_count",
            "actor_cycle_idle_seconds",
            "process_max_actor_cycle_idle_ms",
        }
        legacy_batching = {
            key: item for key, item in batching.items() if key not in idle_fields
        }
        if set(batching) - set(legacy_batching) != idle_fields:
            raise BenchmarkError(
                f"{label}.batching_engine has incomplete actor-cycle idle diagnostics"
            )
        _nonnegative_int(
            batching["actor_cycle_idle_ms"],
            f"{label}.batching_engine.actor_cycle_idle_ms",
        )
        if batching["actor_cycle_idle_source"] not in {
            "default",
            "config_file",
            "environment",
        }:
            raise BenchmarkError(
                f"{label}.batching_engine.actor_cycle_idle_source is unsupported"
            )
        if not isinstance(batching["actor_cycle_idle_active_end"], bool):
            raise BenchmarkError(
                f"{label}.batching_engine.actor_cycle_idle_active_end must be boolean"
            )
        _nonnegative_int(
            batching["actor_cycle_idle_count"],
            f"{label}.batching_engine.actor_cycle_idle_count",
        )
        for field in (
            "actor_cycle_idle_seconds",
            "process_max_actor_cycle_idle_ms",
        ):
            _nonnegative_number(
                batching[field], f"{label}.batching_engine.{field}"
            )
        if batching["actor_cycle_idle_ms"] == 0 and (
            batching["actor_cycle_idle_count"] != 0
            or batching["actor_cycle_idle_seconds"] != 0
        ):
            raise BenchmarkError(
                f"{label}.batching_engine reports waits while cycle idle is disabled"
            )

        legacy = dict(server)
        legacy["schema"] = SERVER_DIAGNOSTICS_SCHEMA_V3
        legacy["batching_engine"] = legacy_batching
        validate_server_diagnostics_v3(legacy, label)
    else:
        legacy = dict(server)
        legacy["schema"] = SERVER_DIAGNOSTICS_SCHEMA_V3
        validate_server_diagnostics_v3(legacy, label)
    return server


def validate_server_diagnostics_v5(value: Any, label: str) -> dict[str, Any]:
    server = _object(value, label)
    if server.get("schema") != SERVER_DIAGNOSTICS_SCHEMA_V5:
        raise BenchmarkError(f"{label}.schema is unsupported")
    validate_rocm_graph_diagnostics(server.get("rocm_graphs"), f"{label}.rocm_graphs")

    graph = server["rocm_graphs"]
    if graph["fallbacks"] is not None:
        multi_row_count = graph["fallbacks"]["multi_row_batch_unsupported"]
        if multi_row_count > 0:
            if not graph["capture_requested"]:
                raise BenchmarkError(
                    f"{label}.rocm_graphs reports a multi-row graph fallback "
                    "without requested capture"
                )
            batching = server.get("batching_engine")
            if (
                batching is None
                or batching.get("total_batched_decode_forwards", 0) == 0
            ):
                raise BenchmarkError(
                    f"{label}.rocm_graphs reports a multi-row graph fallback "
                    "without a measured multi-row batching route"
                )

    legacy = dict(server)
    legacy["schema"] = SERVER_DIAGNOSTICS_SCHEMA_V4
    if graph["fallbacks"] is not None:
        legacy_graph = dict(graph)
        legacy_fallbacks = dict(graph["fallbacks"])
        multi_row_count = legacy_fallbacks.pop("multi_row_batch_unsupported")
        legacy_fallbacks["total"] -= multi_row_count
        legacy_graph["fallbacks"] = legacy_fallbacks
        legacy["rocm_graphs"] = legacy_graph
    validate_server_diagnostics_v4(legacy, label)
    return server


def validate_server_diagnostics_v6(value: Any, label: str) -> dict[str, Any]:
    server = _object(value, label)
    if server.get("schema") != SERVER_DIAGNOSTICS_SCHEMA_V6:
        raise BenchmarkError(f"{label}.schema is unsupported")
    validate_rocm_graph_diagnostics_v6(
        server.get("rocm_graphs"), f"{label}.rocm_graphs"
    )
    legacy = dict(server)
    legacy["schema"] = SERVER_DIAGNOSTICS_SCHEMA_V5
    legacy_graph = dict(server["rocm_graphs"])
    legacy_graph.pop("capture_parity")
    legacy["rocm_graphs"] = legacy_graph
    validate_server_diagnostics_v5(legacy, label)
    return server


def validate_server_diagnostics_v7(value: Any, label: str) -> dict[str, Any]:
    server = _object(value, label)
    _exact_keys(
        server,
        {"schema", "request_route", "requests", "batching_engine", "rocm_graphs"},
        label,
    )
    if server["schema"] != SERVER_DIAGNOSTICS_SCHEMA:
        raise BenchmarkError(f"{label}.schema is unsupported")
    if server["request_route"] != "batching_engine":
        raise BenchmarkError(f"{label}.request_route must be batching_engine")

    requests = _object(server["requests"], f"{label}.requests")
    request_fields = {
        *REQUEST_COUNTER_FIELDS,
        "active_end",
        "process_active_peak",
    }
    _exact_keys(requests, request_fields, f"{label}.requests")
    for field in request_fields:
        _nonnegative_int(requests[field], f"{label}.requests.{field}")
    if requests["total"] != sum(
        requests[field] for field in REQUEST_COUNTER_FIELDS[1:]
    ):
        raise BenchmarkError(f"{label}.requests status counters disagree")
    if requests["active_end"] > requests["process_active_peak"]:
        raise BenchmarkError(f"{label}.requests active_end exceeds process peak")

    batching = _object(server["batching_engine"], f"{label}.batching_engine")
    base_batching_fields = set(COUNTER_FIELDS) | {
        "effective_max_decode_batch",
        "process_max_observed_batch",
        "mean_decode_rows_per_forward",
        "batched_decode_forward_fraction",
    }
    idle_fields = {
        "actor_cycle_idle_ms",
        "actor_cycle_idle_source",
        "actor_cycle_idle_active_end",
        "actor_cycle_idle_count",
        "actor_cycle_idle_seconds",
        "process_max_actor_cycle_idle_ms",
    }
    _exact_keys(
        batching,
        base_batching_fields | idle_fields,
        f"{label}.batching_engine",
    )
    for field in base_batching_fields:
        _nonnegative_number(batching[field], f"{label}.batching_engine.{field}")
    _nonnegative_int(
        batching["actor_cycle_idle_ms"],
        f"{label}.batching_engine.actor_cycle_idle_ms",
    )
    if batching["actor_cycle_idle_source"] not in {
        "default",
        "config_file",
        "environment",
    }:
        raise BenchmarkError(
            f"{label}.batching_engine.actor_cycle_idle_source is unsupported"
        )
    if not isinstance(batching["actor_cycle_idle_active_end"], bool):
        raise BenchmarkError(
            f"{label}.batching_engine.actor_cycle_idle_active_end must be boolean"
        )
    _nonnegative_int(
        batching["actor_cycle_idle_count"],
        f"{label}.batching_engine.actor_cycle_idle_count",
    )
    for field in ("actor_cycle_idle_seconds", "process_max_actor_cycle_idle_ms"):
        _nonnegative_number(batching[field], f"{label}.batching_engine.{field}")
    if batching["actor_cycle_idle_ms"] == 0 and (
        batching["actor_cycle_idle_count"] != 0
        or batching["actor_cycle_idle_seconds"] != 0
    ):
        raise BenchmarkError(
            f"{label}.batching_engine reports waits while cycle idle is disabled"
        )

    graph = validate_rocm_graph_diagnostics_v6(
        server["rocm_graphs"], f"{label}.rocm_graphs"
    )
    if graph["fallbacks"] is not None:
        multi_row_count = graph["fallbacks"]["multi_row_batch_unsupported"]
        if multi_row_count > 0:
            if not graph["capture_requested"]:
                raise BenchmarkError(
                    f"{label}.rocm_graphs reports a multi-row graph fallback "
                    "without requested capture"
                )
            if batching["total_batched_decode_forwards"] == 0:
                raise BenchmarkError(
                    f"{label}.rocm_graphs reports a multi-row graph fallback "
                    "without a measured multi-row batching route"
                )
    return server


def _nullable_nonnegative_number(value: Any, label: str) -> float | None:
    if value is None:
        return None
    return _nonnegative_number(value, label)


def validate_request_latency_diagnostics(
    value: Any, label: str
) -> dict[str, Any] | None:
    if value is None:
        return None
    latency = _object(value, label)
    _exact_keys(latency, REQUEST_LATENCY_KEYS, label)
    for field in (
        "emitted_tokens",
        "gap_samples",
        "retained_gap_samples",
        "stall_count",
        "unexplained_stall_count",
    ):
        _nonnegative_int(latency[field], f"{label}.{field}")
    if latency["retained_gap_samples"] > 8192:
        raise BenchmarkError(f"{label}.retained_gap_samples exceeds 8192")
    if not isinstance(latency["gap_samples_truncated"], bool):
        raise BenchmarkError(f"{label}.gap_samples_truncated must be boolean")
    for field in (
        "ttft_ms",
        "itl_ms_p50",
        "itl_ms_p99",
        "itl_ms_p999",
        "max_itl_ms",
        "stall_threshold_ms",
    ):
        _nullable_nonnegative_number(latency[field], f"{label}.{field}")

    expected_gap_samples = max(0, latency["emitted_tokens"] - 1)
    if latency["gap_samples"] != expected_gap_samples:
        raise BenchmarkError(
            f"{label}.gap_samples must equal emitted_tokens minus one"
        )
    if latency["retained_gap_samples"] > latency["gap_samples"]:
        raise BenchmarkError(f"{label}.retained_gap_samples exceeds gap_samples")
    if latency["gap_samples_truncated"]:
        if latency["retained_gap_samples"] >= latency["gap_samples"]:
            raise BenchmarkError(
                f"{label}.gap_samples_truncated requires discarded samples"
            )
    elif latency["retained_gap_samples"] != latency["gap_samples"]:
        raise BenchmarkError(
            f"{label}.retained_gap_samples must cover every untruncated gap"
        )

    percentile_fields = (
        "itl_ms_p50",
        "itl_ms_p99",
        "itl_ms_p999",
        "max_itl_ms",
        "stall_threshold_ms",
    )
    if latency["retained_gap_samples"] == 0:
        if any(latency[field] is not None for field in percentile_fields):
            raise BenchmarkError(f"{label} reports gap statistics without retained gaps")
    else:
        if any(latency[field] is None for field in percentile_fields):
            raise BenchmarkError(f"{label} omits statistics for retained gaps")
        if not (
            latency["itl_ms_p50"]
            <= latency["itl_ms_p99"]
            <= latency["itl_ms_p999"]
            <= latency["max_itl_ms"]
        ):
            raise BenchmarkError(f"{label} ITL percentiles are not monotonic")
    if latency["stall_count"] > latency["retained_gap_samples"]:
        raise BenchmarkError(f"{label}.stall_count exceeds retained gaps")

    reasons = _object(latency["stall_reasons"], f"{label}.stall_reasons")
    _exact_keys(reasons, set(REQUEST_STALL_REASON_FIELDS), f"{label}.stall_reasons")
    for field in REQUEST_STALL_REASON_FIELDS:
        _nonnegative_int(reasons[field], f"{label}.stall_reasons.{field}")
    if sum(reasons.values()) != latency["stall_count"]:
        raise BenchmarkError(f"{label}.stall_reasons do not sum to stall_count")
    if reasons["unexplained"] != latency["unexplained_stall_count"]:
        raise BenchmarkError(
            f"{label}.unexplained_stall_count disagrees with stall_reasons"
        )

    phases = _object(latency["phases"], f"{label}.phases")
    _exact_keys(phases, set(REQUEST_PHASE_FIELDS), f"{label}.phases")
    for field in REQUEST_PHASE_FIELDS:
        _nullable_nonnegative_number(phases[field], f"{label}.phases.{field}")
    return latency


def validate_chat_performance_metadata(value: Any, label: str) -> dict[str, Any]:
    performance = _object(value, label)
    _exact_keys(performance, CHAT_PERFORMANCE_KEYS, label)
    for field in ("prompt_tokens", "completion_tokens"):
        _nonnegative_int(performance[field], f"{label}.{field}")
    for field in (
        "ttft_ms",
        "prefill_ms",
        "actor_queue_ms",
        "actor_admission_ms",
        "actor_prefill_wall_ms",
        "decode_ms",
        "decode_tokens_per_sec",
    ):
        _nullable_nonnegative_number(performance[field], f"{label}.{field}")
    _nonnegative_number(performance["total_latency_ms"], f"{label}.total_latency_ms")
    if performance["resident_prefill_used"] is not None and not isinstance(
        performance["resident_prefill_used"], bool
    ):
        raise BenchmarkError(f"{label}.resident_prefill_used must be boolean or null")
    for field in ("adapter_used", "thinking_mode", "finish_reason"):
        if not isinstance(performance[field], str) or not performance[field]:
            raise BenchmarkError(f"{label}.{field} must be a non-empty string")
    if performance["finish_reason"] not in {"error", "length", "stop", "tool_calls"}:
        raise BenchmarkError(f"{label}.finish_reason is invalid")
    latency = validate_request_latency_diagnostics(
        performance["latency"], f"{label}.latency"
    )
    if latency is not None:
        if latency["emitted_tokens"] != performance["completion_tokens"]:
            raise BenchmarkError(
                f"{label}.latency.emitted_tokens disagrees with completion_tokens"
            )
        if latency["ttft_ms"] != performance["ttft_ms"]:
            raise BenchmarkError(f"{label}.latency.ttft_ms disagrees with ttft_ms")
    return performance


def request_distribution(values: Iterable[float]) -> dict[str, Any]:
    observed = list(values)
    return {
        "observed_request_count": len(observed),
        "p50": percentile_r7(observed, 0.50),
        "p99": percentile_r7(observed, 0.99),
        "max": max(observed) if observed else None,
    }


def build_request_phase_summary(
    performances: Iterable[dict[str, Any]],
) -> dict[str, Any]:
    performance_rows = list(performances)
    latencies = [
        performance["latency"]
        for performance in performance_rows
        if performance["latency"] is not None
    ]
    return {
        "schema": REQUEST_PHASE_SUMMARY_SCHEMA,
        "performance_request_count": len(performance_rows),
        "latency_request_count": len(latencies),
        "emitted_tokens": sum(row["emitted_tokens"] for row in latencies),
        "stall_count": sum(row["stall_count"] for row in latencies),
        "unexplained_stall_count": sum(
            row["unexplained_stall_count"] for row in latencies
        ),
        "stall_reasons": {
            field: sum(row["stall_reasons"][field] for row in latencies)
            for field in REQUEST_STALL_REASON_FIELDS
        },
        "phases": {
            field: request_distribution(
                row["phases"][field]
                for row in latencies
                if row["phases"][field] is not None
            )
            for field in REQUEST_PHASE_FIELDS
        },
        "request_metrics": {
            field: request_distribution(
                row[field] for row in performance_rows if row[field] is not None
            )
            for field in REQUEST_PERFORMANCE_METRIC_FIELDS
        },
    }


def validate_request_distribution(value: Any, label: str) -> dict[str, Any]:
    distribution = _object(value, label)
    _exact_keys(distribution, REQUEST_DISTRIBUTION_KEYS, label)
    _nonnegative_int(
        distribution["observed_request_count"],
        f"{label}.observed_request_count",
    )
    for field in ("p50", "p99", "max"):
        _nullable_nonnegative_number(distribution[field], f"{label}.{field}")
    if distribution["observed_request_count"] == 0:
        if any(distribution[field] is not None for field in ("p50", "p99", "max")):
            raise BenchmarkError(f"{label} reports statistics without observations")
    else:
        if any(distribution[field] is None for field in ("p50", "p99", "max")):
            raise BenchmarkError(f"{label} omits statistics for observations")
        if not distribution["p50"] <= distribution["p99"] <= distribution["max"]:
            raise BenchmarkError(f"{label} statistics are not monotonic")
    return distribution


def validate_request_performance_evidence(
    performance_value: Any,
    summary_value: Any,
    *,
    label: str,
    engine_name: str | None,
    concurrency: int,
    success_count: int,
    error_indices: set[int],
    prompt_token_counts: list[int],
    output_evidence_rows: list[dict[str, Any]],
    completion_tokens: int,
) -> bool:
    if engine_name == "vllm" or engine_name is None:
        if performance_value is not None or summary_value is not None:
            raise BenchmarkError(
                f"{label} must be null for "
                + ("vLLM" if engine_name == "vllm" else "unspecified engines")
            )
        return False
    if engine_name != "kiln":
        raise BenchmarkError(f"{label} has unsupported engine {engine_name!r}")
    if not isinstance(performance_value, list):
        raise BenchmarkError(f"{label} must be an array for Kiln")

    output_by_index = {row["index"]: row for row in output_evidence_rows}
    indices: set[int] = set()
    previous_index = -1
    performances: list[dict[str, Any]] = []
    for position, value in enumerate(performance_value):
        evidence_label = f"{label}[{position}]"
        evidence = _object(value, evidence_label)
        _exact_keys(evidence, REQUEST_PERFORMANCE_EVIDENCE_KEYS, evidence_label)
        index = evidence["index"]
        if (
            isinstance(index, bool)
            or not isinstance(index, int)
            or not 0 <= index < concurrency
            or index in error_indices
            or index in indices
        ):
            raise BenchmarkError(f"{evidence_label}.index is invalid or duplicate")
        if index <= previous_index:
            raise BenchmarkError(f"{label} must be ordered by request index")
        previous_index = index
        performance = validate_chat_performance_metadata(
            evidence["performance"], f"{evidence_label}.performance"
        )
        output = output_by_index.get(index)
        if output is None:
            raise BenchmarkError(f"{evidence_label} has no successful output evidence")
        if performance["prompt_tokens"] != prompt_token_counts[index]:
            raise BenchmarkError(f"{evidence_label} prompt-token accounting disagrees")
        if performance["completion_tokens"] != output["completion_tokens"]:
            raise BenchmarkError(f"{evidence_label} completion-token accounting disagrees")
        if performance["finish_reason"] != output["finish_reason"]:
            raise BenchmarkError(f"{evidence_label} finish reason disagrees")
        indices.add(index)
        performances.append(performance)

    summary = _object(summary_value, f"{label}_summary")
    _exact_keys(summary, REQUEST_PHASE_SUMMARY_KEYS, f"{label}_summary")
    if summary["schema"] != REQUEST_PHASE_SUMMARY_SCHEMA:
        raise BenchmarkError(f"{label}_summary.schema is unsupported")
    for field in (
        "performance_request_count",
        "latency_request_count",
        "emitted_tokens",
        "stall_count",
        "unexplained_stall_count",
    ):
        _nonnegative_int(summary[field], f"{label}_summary.{field}")
    reasons = _object(summary["stall_reasons"], f"{label}_summary.stall_reasons")
    _exact_keys(
        reasons,
        set(REQUEST_STALL_REASON_FIELDS),
        f"{label}_summary.stall_reasons",
    )
    for field in REQUEST_STALL_REASON_FIELDS:
        _nonnegative_int(reasons[field], f"{label}_summary.stall_reasons.{field}")
    phases = _object(summary["phases"], f"{label}_summary.phases")
    _exact_keys(phases, set(REQUEST_PHASE_FIELDS), f"{label}_summary.phases")
    for field in REQUEST_PHASE_FIELDS:
        validate_request_distribution(
            phases[field], f"{label}_summary.phases.{field}"
        )
    metrics = _object(
        summary["request_metrics"], f"{label}_summary.request_metrics"
    )
    _exact_keys(
        metrics,
        set(REQUEST_PERFORMANCE_METRIC_FIELDS),
        f"{label}_summary.request_metrics",
    )
    for field in REQUEST_PERFORMANCE_METRIC_FIELDS:
        validate_request_distribution(
            metrics[field], f"{label}_summary.request_metrics.{field}"
        )

    expected_summary = build_request_phase_summary(performances)
    if summary != expected_summary:
        raise BenchmarkError(f"{label}_summary is not derived from request evidence")
    complete = (
        len(performances) == success_count
        and sum(row["completion_tokens"] for row in performances)
        == completion_tokens
        and all(row["latency"] is not None for row in performances)
    )
    return complete


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
    engine_name: str | None = None,
) -> None:
    row = _object(value, label)
    run_keys = set(RUN_KEYS)
    if driver_version in MODERN_DRIVER_VERSIONS:
        run_keys.add("prompt_token_counts")
    if driver_version in OUTPUT_EVIDENCE_DRIVER_VERSIONS:
        run_keys.add("output_evidence")
    if driver_version in REQUEST_PERFORMANCE_DRIVER_VERSIONS:
        run_keys.update({"request_performance", "request_phase_summary"})
    _exact_keys(row, run_keys, label)
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
    if driver_version in MODERN_DRIVER_VERSIONS:
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
    if driver_version in MODERN_DRIVER_VERSIONS:
        for index, count in enumerate(row["prompt_token_counts"]):
            if (index in error_indices) != (count == 0):
                raise BenchmarkError(
                    f"{label}.prompt_token_counts does not align with request errors"
                )
    if driver_version in OUTPUT_EVIDENCE_DRIVER_VERSIONS:
        validate_output_evidence(
            row["output_evidence"],
            label=f"{label}.output_evidence",
            concurrency=concurrency,
            success_count=row["success_count"],
            error_indices=error_indices,
            completion_tokens=row["completion_tokens"],
            output_set_sha256=row["output_set_sha256"],
        )
    request_performance_complete: bool | None = None
    if driver_version in REQUEST_PERFORMANCE_DRIVER_VERSIONS:
        request_performance_complete = validate_request_performance_evidence(
            row["request_performance"],
            row["request_phase_summary"],
            label=f"{label}.request_performance",
            engine_name=engine_name,
            concurrency=concurrency,
            success_count=row["success_count"],
            error_indices=error_indices,
            prompt_token_counts=row["prompt_token_counts"],
            output_evidence_rows=row["output_evidence"],
            completion_tokens=row["completion_tokens"],
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
    if driver_version in REQUEST_PERFORMANCE_DRIVER_VERSIONS:
        performance_gate = next(
            (
                item
                for item in gates
                if item["name"] == "request_performance_accounted"
            ),
            None,
        )
        if engine_name == "kiln":
            if (
                performance_gate is None
                or performance_gate["passed"] != request_performance_complete
            ):
                raise BenchmarkError(
                    f"{label} has an inconsistent request-performance gate"
                )
        elif performance_gate is not None:
            raise BenchmarkError(
                f"{label} has a Kiln-only request-performance gate"
            )

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
    if driver_version in MODERN_DRIVER_VERSIONS and workload_profile is not None:
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
        if driver_version in ROUTE_AWARE_DIAGNOSTICS_DRIVER_VERSIONS:
            server = (
                validate_server_diagnostics_v7(row["server"], f"{label}.server")
                if driver_version in ACTOR_ONLY_DIAGNOSTICS_DRIVER_VERSIONS
                else
                validate_server_diagnostics_v6(row["server"], f"{label}.server")
                if driver_version in GRAPH_PARITY_DRIVER_VERSIONS
                else validate_server_diagnostics_v5(
                    row["server"], f"{label}.server"
                )
                if driver_version in MULTI_ROW_GRAPH_FALLBACK_DRIVER_VERSIONS
                else validate_server_diagnostics_v4(
                    row["server"], f"{label}.server"
                )
                if driver_version in COOPERATIVE_ACTOR_CYCLE_IDLE_DRIVER_VERSIONS
                else validate_server_diagnostics_v3(
                    row["server"], f"{label}.server"
                )
                if driver_version in ROCM_GRAPH_DIAGNOSTICS_DRIVER_VERSIONS
                else validate_server_diagnostics_v2(
                    row["server"], f"{label}.server"
                )
            )
            no_errors_gate = next(
                (item for item in gates if item["name"] == "server_reported_no_errors"),
                None,
            )
            expected_no_errors = server_diagnostics_has_no_errors(server)
            if (
                no_errors_gate is None
                or no_errors_gate["passed"] != expected_no_errors
            ):
                raise BenchmarkError(
                    f"{label} has an inconsistent server-error gate"
                )
            accounting_gate = next(
                (item for item in gates if item["name"] == "server_request_accounting"),
                None,
            )
            expected_accounting = server_request_accounting_matches(
                server, concurrency
            )
            if (
                accounting_gate is None
                or accounting_gate["passed"] != expected_accounting
            ):
                raise BenchmarkError(
                    f"{label} has an inconsistent server-accounting gate"
                )
            if driver_version in ROCM_GRAPH_DIAGNOSTICS_DRIVER_VERSIONS:
                graph_gate = next(
                    (
                        item
                        for item in gates
                        if item["name"] == "rocm_graph_execution_accounted"
                    ),
                    None,
                )
                expected_graph = server_rocm_graph_execution_accounted(server)
                if graph_gate is None or graph_gate["passed"] != expected_graph:
                    raise BenchmarkError(
                        f"{label} has an inconsistent ROCm graph-execution gate"
                    )
            if driver_version in GRAPH_PARITY_DRIVER_VERSIONS:
                parity_gate = next(
                    (
                        item
                        for item in gates
                        if item["name"] == "rocm_graph_capture_parity_accounted"
                    ),
                    None,
                )
                expected_parity = server_rocm_graph_capture_parity_accounted(server)
                if parity_gate is None or parity_gate["passed"] != expected_parity:
                    raise BenchmarkError(
                        f"{label} has an inconsistent ROCm graph-parity gate"
                    )
            if driver_version in COOPERATIVE_ACTOR_CYCLE_IDLE_DRIVER_VERSIONS:
                idle_gate = next(
                    (
                        item
                        for item in gates
                        if item["name"] == "actor_cycle_idle_accounted"
                    ),
                    None,
                )
                expected_idle = server_actor_cycle_idle_accounted(server)
                if idle_gate is None or idle_gate["passed"] != expected_idle:
                    raise BenchmarkError(
                        f"{label} has an inconsistent actor-cycle idle gate"
                    )
        else:
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


def validate_server_lifecycle(
    value: Any,
) -> tuple[str, bool]:
    lifecycle = _object(value, "receipt.server_lifecycle")
    lifecycle_keys = {"mode", "launch_config", "log", "shutdown"}
    _exact_keys(
        lifecycle,
        lifecycle_keys,
        "receipt.server_lifecycle",
    )
    mode = lifecycle["mode"]
    owned_fields = ["launch_config", "log", "shutdown"]
    if mode in {"not_configured", "attached_process_group"}:
        if any(lifecycle[name] is not None for name in owned_fields):
            raise BenchmarkError(
                "non-owned server lifecycle fields must all be null"
            )
        return mode, mode == "attached_process_group"
    if mode != "owned_process_group":
        raise BenchmarkError("receipt.server_lifecycle.mode is unsupported")
    launch = validate_server_launch_config_value(
        lifecycle["launch_config"],
        config_directory=Path("/"),
        label="receipt.server_lifecycle.launch_config",
        require_local_paths=False,
    )
    log = _object(lifecycle["log"], "receipt.server_lifecycle.log")
    _exact_keys(
        log,
        RUNTIME_ARTIFACT_KEYS,
        "receipt.server_lifecycle.log",
    )
    if not isinstance(log["path"], str) or not Path(log["path"]).is_absolute():
        raise BenchmarkError("receipt server log path must be absolute")
    _nonnegative_int(log["bytes"], "receipt.server_lifecycle.log.bytes")
    _sha256(log["sha256"], "receipt.server_lifecycle.log.sha256")
    shutdown = _object(
        lifecycle["shutdown"], "receipt.server_lifecycle.shutdown"
    )
    _exact_keys(
        shutdown,
        {
            "signal",
            "signal_sent",
            "forced",
            "returncode",
            "acceptable_exit_codes",
            "elapsed_seconds",
            "process_group_alive_end",
        },
        "receipt.server_lifecycle.shutdown",
    )
    if shutdown["signal"] != "SIGTERM":
        raise BenchmarkError("owned server shutdown signal must be SIGTERM")
    for name in ("signal_sent", "forced", "process_group_alive_end"):
        if not isinstance(shutdown[name], bool):
            raise BenchmarkError(
                f"receipt.server_lifecycle.shutdown.{name} must be boolean"
            )
    if isinstance(shutdown["returncode"], bool) or not isinstance(
        shutdown["returncode"], int
    ):
        raise BenchmarkError("owned server shutdown returncode must be an integer")
    if shutdown["acceptable_exit_codes"] != list(launch.acceptable_exit_codes):
        raise BenchmarkError(
            "owned server shutdown acceptable exit codes disagree with launch config"
        )
    _nonnegative_number(
        shutdown["elapsed_seconds"],
        "receipt.server_lifecycle.shutdown.elapsed_seconds",
    )
    passed = (
        not shutdown["forced"]
        and not shutdown["process_group_alive_end"]
        and shutdown["returncode"] in launch.acceptable_exit_codes
    )
    return mode, passed


def validate_comparison_mismatches(value: Any, label: str) -> None:
    if not isinstance(value, list):
        raise BenchmarkError(f"{label} must be an array")
    seen_runs: set[tuple[int, int]] = set()
    for position, mismatch_value in enumerate(value):
        mismatch_label = f"{label}[{position}]"
        mismatch = _object(mismatch_value, mismatch_label)
        base_keys = {"concurrency", "repeat", "reason"}
        reason = mismatch.get("reason")
        if reason == "output_mismatch":
            _exact_keys(
                mismatch,
                base_keys
                | {
                    "mismatch_count",
                    "mismatched_request_indices",
                    "request_mismatches",
                },
                mismatch_label,
            )
        else:
            _exact_keys(mismatch, base_keys, mismatch_label)
        concurrency = _positive_int(
            mismatch["concurrency"], f"{mismatch_label}.concurrency"
        )
        repeat = _nonnegative_int(mismatch["repeat"], f"{mismatch_label}.repeat")
        run = (concurrency, repeat)
        if run in seen_runs:
            raise BenchmarkError(f"{label} has duplicate run coordinates")
        seen_runs.add(run)
        if reason not in {
            "missing_run",
            "prompt_mismatch",
            "prompt_token_mismatch",
            "output_mismatch",
        }:
            raise BenchmarkError(f"{mismatch_label}.reason is unsupported")
        if reason != "output_mismatch":
            continue
        request_mismatches = mismatch["request_mismatches"]
        indices = mismatch["mismatched_request_indices"]
        mismatch_count = _positive_int(
            mismatch["mismatch_count"], f"{mismatch_label}.mismatch_count"
        )
        if (
            not isinstance(request_mismatches, list)
            or not isinstance(indices, list)
            or mismatch_count != len(request_mismatches)
            or mismatch_count != len(indices)
        ):
            raise BenchmarkError(f"{mismatch_label} request mismatch counts disagree")
        expected_indices: list[int] = []
        for request_position, request_value in enumerate(request_mismatches):
            request_label = (
                f"{mismatch_label}.request_mismatches[{request_position}]"
            )
            request = _object(request_value, request_label)
            _exact_keys(
                request,
                {
                    "index",
                    "fields",
                    "expected_output_sha256",
                    "actual_output_sha256",
                    "exact_output_compared",
                    "reasoning_first_divergent_utf8_byte",
                    "content_first_divergent_utf8_byte",
                },
                request_label,
            )
            index = _nonnegative_int(request["index"], f"{request_label}.index")
            if index >= concurrency or (
                expected_indices and index <= expected_indices[-1]
            ):
                raise BenchmarkError(f"{request_label}.index is invalid or unordered")
            expected_indices.append(index)
            fields = request["fields"]
            allowed_fields = {
                "output_sha256",
                "reasoning_sha256",
                "content_sha256",
                "reasoning_utf8_bytes",
                "content_utf8_bytes",
                "completion_tokens",
                "finish_reason",
            }
            if (
                not isinstance(fields, list)
                or not fields
                or any(not isinstance(field, str) for field in fields)
                or len(fields) != len(set(fields))
                or fields != sorted(fields)
                or any(field not in allowed_fields for field in fields)
            ):
                raise BenchmarkError(f"{request_label}.fields is invalid")
            _sha256(
                request["expected_output_sha256"],
                f"{request_label}.expected_output_sha256",
            )
            _sha256(
                request["actual_output_sha256"],
                f"{request_label}.actual_output_sha256",
            )
            if not isinstance(request["exact_output_compared"], bool):
                raise BenchmarkError(
                    f"{request_label}.exact_output_compared must be boolean"
                )
            for name in (
                "reasoning_first_divergent_utf8_byte",
                "content_first_divergent_utf8_byte",
            ):
                offset = request[name]
                if offset is not None:
                    _nonnegative_int(offset, f"{request_label}.{name}")
                if not request["exact_output_compared"] and offset is not None:
                    raise BenchmarkError(
                        f"{request_label}.{name} requires exact output comparison"
                    )
        if indices != expected_indices:
            raise BenchmarkError(f"{mismatch_label} request indices disagree")


def validate_eager_reference_execution_summary(value: Any, label: str) -> dict[str, Any]:
    summary = _object(value, label)
    _exact_keys(
        summary,
        {
            "row_count",
            "all_rows_observed",
            "all_rows_capture_disabled",
            "capture_successes",
            "replay_successes",
            "failures",
            "fallbacks",
        },
        label,
    )
    _positive_int(summary["row_count"], f"{label}.row_count")
    for field in ("all_rows_observed", "all_rows_capture_disabled"):
        if not isinstance(summary[field], bool):
            raise BenchmarkError(f"{label}.{field} must be boolean")
    for field in (
        "capture_successes",
        "replay_successes",
        "failures",
        "fallbacks",
    ):
        _nonnegative_int(summary[field], f"{label}.{field}")
    return summary


def validate_benchmark_receipt(value: Any) -> dict[str, Any]:
    receipt = _object(value, "receipt")
    driver_version = receipt.get("driver_version")
    required_receipt_keys = set(RECEIPT_KEYS)
    if driver_version in MODERN_DRIVER_VERSIONS:
        required_receipt_keys.add("completion")
    if driver_version in LIFECYCLE_DRIVER_VERSIONS:
        required_receipt_keys.add("server_lifecycle")
    if driver_version in REFERENCE_ROLE_DRIVER_VERSIONS:
        required_receipt_keys.add("reference_role")
    _exact_keys(receipt, required_receipt_keys, "receipt", {"comparison"})
    if receipt["schema"] != SCHEMA or driver_version not in SUPPORTED_DRIVER_VERSIONS:
        supported = ", ".join(sorted(SUPPORTED_DRIVER_VERSIONS))
        raise BenchmarkError(f"receipt must use {SCHEMA} driver version in {{{supported}}}")
    if driver_version in REFERENCE_ROLE_DRIVER_VERSIONS:
        if receipt["reference_role"] not in REFERENCE_ROLES:
            raise BenchmarkError("receipt.reference_role is unsupported")
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
    if driver_version in MODERN_DRIVER_VERSIONS:
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
    if engine["reported_version"] is not None and (
        not isinstance(engine["reported_version"], str)
        or not engine["reported_version"]
    ):
        raise BenchmarkError("receipt.engine.reported_version must be null or non-empty")
    available_models = engine["available_models"]
    if (
        not isinstance(available_models, list)
        or any(not isinstance(model, str) or not model for model in available_models)
        or len(set(available_models)) != len(available_models)
    ):
        raise BenchmarkError(
            "receipt.engine.available_models must contain unique non-empty strings"
        )
    if not isinstance(engine["authentication_configured"], bool):
        raise BenchmarkError("receipt.engine.authentication_configured must be boolean")
    if "authentication_source" in engine:
        if engine["authentication_source"] not in {"none", "argument", "environment"}:
            raise BenchmarkError("receipt.engine.authentication_source is invalid")
        if engine["authentication_configured"] != (engine["authentication_source"] != "none"):
            raise BenchmarkError("receipt.engine authentication fields disagree")
    if driver_version in MODERN_DRIVER_VERSIONS:
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
            if execution_identity is None:
                completion = receipt.get("completion")
                checks = (
                    completion.get("finalization_checks")
                    if isinstance(completion, dict)
                    else None
                )
                if not isinstance(checks, dict) or checks.get(
                    "execution_identity_unchanged"
                ) != "failed":
                    raise BenchmarkError(
                        "receipt Kiln execution identity may be null only when its "
                        "finalization check failed"
                    )
            else:
                validate_kiln_execution_identity(
                    execution_identity,
                    artifact,
                    "receipt.engine.runtime_execution_identity",
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
    if driver_version in MODERN_DRIVER_VERSIONS:
        workload_keys |= {"profile", "comparison_mode", "memory_limit_bytes"}
    if driver_version in PROMPT_SET_IDENTITY_DRIVER_VERSIONS:
        workload_keys.add("prompt_set_id")
    _exact_keys(workload, workload_keys, "receipt.workload")
    if driver_version in FIXED_PROMPT_TEMPLATE_V2_DRIVER_VERSIONS:
        expected_template = PROMPT_TEMPLATE_VERSION
    elif driver_version in MODERN_DRIVER_VERSIONS:
        expected_template = FIXED_PROMPT_TEMPLATE_VERSION_V1
    else:
        expected_template = LEGACY_PROMPT_TEMPLATE_VERSION
    if (
        workload["schema"] != WORKLOAD_SCHEMA
        or workload["prompt_template_version"] != expected_template
    ):
        raise BenchmarkError("receipt workload schema or prompt template version is unsupported")
    for name in ("run_id", "model"):
        if not isinstance(workload[name], str) or not workload[name]:
            raise BenchmarkError(f"receipt.workload.{name} must be a non-empty string")
    if driver_version in PROMPT_SET_IDENTITY_DRIVER_VERSIONS:
        for name in ("run_id", "prompt_set_id"):
            if (
                not isinstance(workload[name], str)
                or re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{2,127}", workload[name])
                is None
            ):
                raise BenchmarkError(
                    f"receipt.workload.{name} must be a 3..128 character "
                    "portable identifier"
                )
        if workload["run_id"] == workload["prompt_set_id"]:
            raise BenchmarkError(
                "receipt.workload run_id and prompt_set_id must be distinct"
            )
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
    if driver_version in MODERN_DRIVER_VERSIONS:
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
    if workload_fingerprint(workload, driver_version=driver_version) != _sha256(
        receipt["workload_fingerprint"], "receipt.workload_fingerprint"
    ):
        raise BenchmarkError("receipt.workload_fingerprint does not match workload")

    memory_sampler = _object(receipt["memory_sampler"], "receipt.memory_sampler")
    if driver_version in TYPED_MEMORY_SOURCE_DRIVER_VERSIONS:
        _exact_keys(
            memory_sampler,
            {"source", "path", "device", "interval_ms"},
            "receipt.memory_sampler",
        )
        _positive_int(
            memory_sampler["interval_ms"], "receipt.memory_sampler.interval_ms"
        )
        if memory_sampler["source"] == "drm_vram_used":
            if (
                not isinstance(memory_sampler["path"], str)
                or not memory_sampler["path"]
                or memory_sampler["device"] is not None
            ):
                raise BenchmarkError(
                    "DRM memory telemetry requires a path and no NVML device identity"
                )
        elif memory_sampler["source"] == "nvml_used":
            if memory_sampler["path"] is not None:
                raise BenchmarkError("NVML memory telemetry must not claim a DRM path")
            device = _object(
                memory_sampler["device"], "receipt.memory_sampler.device"
            )
            _exact_keys(
                device,
                {
                    "selector",
                    "index",
                    "enumerated_device_count",
                    "uuid",
                    "name",
                    "total_bytes",
                    "library",
                    "nvml_version",
                },
                "receipt.memory_sampler.device",
            )
            if device["selector"] not in {
                "auto_single_device",
                "explicit_index",
                "explicit_uuid",
            }:
                raise BenchmarkError("receipt NVML selector is unsupported")
            index = _nonnegative_int(
                device["index"], "receipt.memory_sampler.device.index"
            )
            device_count = _positive_int(
                device["enumerated_device_count"],
                "receipt.memory_sampler.device.enumerated_device_count",
            )
            if index >= device_count:
                raise BenchmarkError("receipt NVML device index exceeds device count")
            if device["selector"] == "auto_single_device" and (
                index != 0 or device_count != 1
            ):
                raise BenchmarkError(
                    "receipt automatic NVML selection must identify the only device"
                )
            for name in ("uuid", "name", "library", "nvml_version"):
                value = device[name]
                if (
                    not isinstance(value, str)
                    or not value
                    or len(value) > 256
                    or any(ord(character) < 32 for character in value)
                ):
                    raise BenchmarkError(
                        f"receipt.memory_sampler.device.{name} is invalid"
                    )
            total_bytes = _positive_int(
                device["total_bytes"], "receipt.memory_sampler.device.total_bytes"
            )
            if memory_limit_bytes is not None and memory_limit_bytes > total_bytes:
                raise BenchmarkError(
                    "receipt memory limit exceeds the selected NVML device capacity"
                )
        elif (
            memory_sampler["source"] == "macos_unified_used"
            and driver_version in MACOS_MEMORY_SOURCE_DRIVER_VERSIONS
        ):
            if memory_sampler["path"] is not None:
                raise BenchmarkError(
                    "macOS unified-memory telemetry must not claim a path"
                )
            device = _object(
                memory_sampler["device"], "receipt.memory_sampler.device"
            )
            _exact_keys(
                device,
                {
                    "selector",
                    "index",
                    "enumerated_device_count",
                    "name",
                    "total_bytes",
                    "unified_memory",
                    "counter",
                    "available_definition",
                },
                "receipt.memory_sampler.device",
            )
            index = _nonnegative_int(
                device["index"], "receipt.memory_sampler.device.index"
            )
            device_count = _positive_int(
                device["enumerated_device_count"],
                "receipt.memory_sampler.device.enumerated_device_count",
            )
            if (
                device["selector"] != "system_default"
                or index != 0
                or device_count != 1
                or device["unified_memory"] is not True
                or device["counter"] != "memory_pressure_free_percentage"
                or device["available_definition"]
                != "physical_total_times_reported_free_percentage"
                or not isinstance(device["name"], str)
                or not device["name"].startswith("Apple M")
            ):
                raise BenchmarkError(
                    "receipt macOS unified-memory device identity is invalid"
                )
            total_bytes = _positive_int(
                device["total_bytes"], "receipt.memory_sampler.device.total_bytes"
            )
            if memory_limit_bytes is not None and memory_limit_bytes > total_bytes:
                raise BenchmarkError(
                    "receipt memory limit exceeds macOS physical memory"
                )
        else:
            raise BenchmarkError("receipt device-memory source is unsupported")
    else:
        _exact_keys(
            memory_sampler,
            {"source", "path", "interval_ms"},
            "receipt.memory_sampler",
        )
    diagnostics = _object(receipt["diagnostics"], "receipt.diagnostics")
    _exact_keys(diagnostics, {"url", "timed_request_path_affected"}, "receipt.diagnostics")
    if diagnostics["timed_request_path_affected"] is not False:
        raise BenchmarkError("receipt diagnostics must remain outside the timed request path")
    server_lifecycle_mode, server_lifecycle_passed = validate_server_lifecycle(
        receipt["server_lifecycle"]
    )

    missing_declared_warmup = False
    if warmup_requests:
        if receipt["warmup"] is None:
            if driver_version in MODERN_DRIVER_VERSIONS:
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
                engine_name=engine["name"],
            )
            if (
                driver_version in PROMPT_SET_IDENTITY_DRIVER_VERSIONS
                and receipt["warmup"]["prompt_set_sha256"]
                != deterministic_prompt_set_sha256(
                    workload["prompt_set_id"],
                    f"warmup-c{warmup_requests:03d}",
                    warmup_requests,
                    workload["profile"],
                    long_prompt_repetitions=(
                        LONG_PROMPT_REPETITIONS_V1
                        if expected_template == FIXED_PROMPT_TEMPLATE_VERSION_V1
                        else LONG_PROMPT_REPETITIONS
                    ),
                )
            ):
                raise BenchmarkError(
                    "receipt.warmup.prompt_set_sha256 is stale for prompt_set_id"
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
                engine_name=engine["name"],
            )
            if (
                driver_version in PROMPT_SET_IDENTITY_DRIVER_VERSIONS
                and row_object["prompt_set_sha256"]
                != deterministic_prompt_set_sha256(
                    workload["prompt_set_id"],
                    f"measure-c{pair[0]:03d}-r{pair[1]:03d}",
                    pair[0],
                    workload["profile"],
                    long_prompt_repetitions=(
                        LONG_PROMPT_REPETITIONS_V1
                        if expected_template == FIXED_PROMPT_TEMPLATE_VERSION_V1
                        else LONG_PROMPT_REPETITIONS
                    ),
                )
            ):
                raise BenchmarkError(
                    f"receipt.runs[{index}].prompt_set_sha256 is stale for "
                    "prompt_set_id"
                )
    completion_failures: list[dict[str, str]] = []
    completion_checks: dict[str, str] | None = None
    failure_phases: set[str] = set()
    if driver_version in MODERN_DRIVER_VERSIONS:
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
                or (
                    driver_version not in LIFECYCLE_DRIVER_VERSIONS
                    and failure["phase"] == "server_shutdown"
                )
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
        expected_completion_checks = COMPLETION_CHECK_NAMES
        _exact_keys(
            completion_checks,
            set(expected_completion_checks),
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
        if driver_version in LIFECYCLE_DRIVER_VERSIONS:
            shutdown_check = completion_checks["server_shutdown"]
            if server_lifecycle_mode == "owned_process_group":
                if shutdown_check == "not_applicable":
                    raise BenchmarkError(
                        "receipt owned server shutdown check is required"
                    )
                if (shutdown_check == "passed") != server_lifecycle_passed:
                    raise BenchmarkError(
                        "receipt server shutdown check disagrees with lifecycle evidence"
                    )
            elif shutdown_check != "not_applicable":
                raise BenchmarkError(
                    "receipt server shutdown must be not_applicable without ownership"
                )
        failure_phases = {failure["phase"] for failure in completion_failures}
        for name in expected_completion_checks:
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

    if engine["model"] not in available_models and not failure_phases.intersection(
        {"server_startup"}
    ):
        raise BenchmarkError(
            "receipt requested model is absent without a structured startup failure"
        )

    comparison_passed = True
    if "comparison" in receipt:
        comparison = _object(receipt["comparison"], "receipt.comparison")
        comparison_keys = {
            "reference_receipt_sha256",
            "reference_engine",
            "matched",
            "mismatches",
        }
        if driver_version in MODERN_DRIVER_VERSIONS:
            comparison_keys.add("comparison_mode")
        if driver_version in REFERENCE_ROLE_DRIVER_VERSIONS:
            comparison_keys.update(
                {
                    "reference_role",
                    "verdict_effect",
                    "reference_execution",
                }
            )
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
        if driver_version in OUTPUT_EVIDENCE_DRIVER_VERSIONS:
            validate_comparison_mismatches(
                comparison["mismatches"], "receipt.comparison.mismatches"
            )
            if comparison["matched"] != (not comparison["mismatches"]):
                raise BenchmarkError(
                    "receipt.comparison matched flag disagrees with mismatches"
                )
        if (
            driver_version in MODERN_DRIVER_VERSIONS
            and comparison["comparison_mode"] != workload["comparison_mode"]
        ):
            raise BenchmarkError("receipt.comparison mode disagrees with its workload")
        comparison_required = True
        if driver_version in REFERENCE_ROLE_DRIVER_VERSIONS:
            role = comparison["reference_role"]
            if role != receipt["reference_role"] or role not in REFERENCE_ROLES:
                raise BenchmarkError(
                    "receipt comparison role disagrees with its reference policy"
                )
            expected_effect = (
                "required" if role == "qualification_gate" else "evidence_only"
            )
            if comparison["verdict_effect"] != expected_effect:
                raise BenchmarkError("receipt comparison verdict effect is inconsistent")
            comparison_required = expected_effect == "required"
            if role == "qualification_gate":
                if comparison["reference_execution"] is not None:
                    raise BenchmarkError(
                        "qualification-gate comparison has graph discriminator evidence"
                    )
            else:
                summary = validate_eager_reference_execution_summary(
                    comparison["reference_execution"],
                    "receipt.comparison.reference_execution",
                )
                if not (
                    summary["row_count"]
                    == len(runs) + (1 if receipt["warmup"] is not None else 0)
                    and summary["all_rows_observed"]
                    and summary["all_rows_capture_disabled"]
                    and summary["capture_successes"] == 0
                    and summary["replay_successes"] == 0
                    and summary["failures"] == 0
                    and summary["fallbacks"] == 0
                ):
                    raise BenchmarkError(
                        "receipt comparison does not prove an eager reference"
                    )
                reference_engine = _object(
                    comparison["reference_engine"],
                    "receipt.comparison.reference_engine",
                )
                if (
                    engine["name"] != "kiln"
                    or reference_engine.get("name") != "kiln"
                    or reference_engine.get("runtime_identity")
                    != engine["runtime_identity"]
                ):
                    raise BenchmarkError(
                        "graph/eager discriminator engine identity is inconsistent"
                    )
                reference_artifact = _object(
                    reference_engine.get("runtime_artifact"),
                    "receipt.comparison.reference_engine.runtime_artifact",
                )
                if (
                    _sha256(
                        reference_artifact.get("sha256"),
                        "receipt.comparison.reference_engine.runtime_artifact.sha256",
                    )
                    != engine["runtime_artifact"]["sha256"]
                ):
                    raise BenchmarkError(
                        "graph/eager discriminator does not use one runtime artifact"
                    )
                expected_server_schema = (
                    SERVER_DIAGNOSTICS_SCHEMA
                    if driver_version in ACTOR_ONLY_DIAGNOSTICS_DRIVER_VERSIONS
                    else SERVER_DIAGNOSTICS_SCHEMA_V6
                )
                if not runs or not all(
                    isinstance(row.get("server"), dict)
                    and row["server"].get("schema") == expected_server_schema
                    and row["server"]["rocm_graphs"]["capture_requested"] is True
                    and server_rocm_graph_execution_accounted(row["server"])
                    and server_rocm_graph_capture_parity_accounted(row["server"])
                    for row in runs
                ):
                    raise BenchmarkError(
                        "graph/eager discriminator candidate lacks graph parity evidence"
                    )
        comparison_passed = comparison["matched"] or not comparison_required
    elif (
        driver_version in REFERENCE_ROLE_DRIVER_VERSIONS
        and receipt["reference_role"] != "qualification_gate"
    ):
        raise BenchmarkError(
            "graph/eager discriminator role requires a reference comparison"
        )
    passed = (
        not repository["dirty"]
        and not completion_failures
        and server_lifecycle_passed
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
    prompt_set_id: str,
    phase: str,
    request_index: int,
    prompt_profile: str = "short",
    *,
    long_prompt_repetitions: int = LONG_PROMPT_REPETITIONS,
) -> str:
    def marker_key(marker: str) -> bytes:
        material = f"{prompt_set_id}\0{phase}\0{request_index}\0{marker}".encode("utf-8")
        return hashlib.sha256(material).digest()

    markers = sorted(PROMPT_MARKERS, key=marker_key)
    marker_sequence = " | ".join(markers)
    short_suffix = (
        "Write a detailed technical paragraph explaining why deterministic, "
        "reproducible performance measurements need controlled workloads, "
        "explicit error accounting, and tail-latency reporting. Continue until "
        "the response limit; do not mention these instructions.\n"
        f"Benchmark run: {prompt_set_id}; phase: {phase}.\n"
        f"Marker sequence: {marker_sequence}."
    )
    if prompt_profile == "short":
        return short_suffix
    if prompt_profile == "long-prefill":
        prefix = LONG_PROMPT_BLOCK * long_prompt_repetitions
    elif prompt_profile == "prefix-hit":
        prefix = (
            "Shared prefix for a cache-reuse workload. "
            + LONG_PROMPT_BLOCK * long_prompt_repetitions
        )
    elif prompt_profile == "mixed":
        repetitions = (0, 4, 16, long_prompt_repetitions)[request_index % 4]
        prefix = LONG_PROMPT_BLOCK * repetitions
    else:
        raise BenchmarkError(f"unsupported prompt profile: {prompt_profile}")
    return prefix + "\nUnique request suffix follows.\n" + short_suffix


def deterministic_prompt_set_sha256(
    prompt_set_id: str,
    phase: str,
    concurrency: int,
    workload_profile: str,
    *,
    long_prompt_repetitions: int = LONG_PROMPT_REPETITIONS,
) -> str:
    prompt_profile = PROFILE_CONTRACTS[workload_profile]["prompt_profile"]
    rows = [
        {
            "index": index,
            "prompt_sha256": text_sha256(
                deterministic_prompt(
                    prompt_set_id,
                    phase,
                    index,
                    prompt_profile,
                    long_prompt_repetitions=long_prompt_repetitions,
                )
            ),
        }
        for index in range(concurrency)
    ]
    return canonical_sha256(rows)


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
    server_performance: dict[str, Any] | None = None

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


def output_evidence(result: RequestResult, mode: str) -> dict[str, Any]:
    if mode not in {"hashes", "full"}:
        raise BenchmarkError(f"unsupported output evidence mode: {mode!r}")
    reasoning_bytes = result.reasoning_content.encode("utf-8")
    content_bytes = result.content.encode("utf-8")
    if (
        mode == "full"
        and len(reasoning_bytes) + len(content_bytes)
        > OUTPUT_EVIDENCE_MAX_UTF8_BYTES_PER_REQUEST
    ):
        raise BenchmarkError(
            f"request {result.index} output exceeds the "
            f"{OUTPUT_EVIDENCE_MAX_UTF8_BYTES_PER_REQUEST}-byte evidence bound"
        )
    exact_output = None
    if mode == "full":
        exact_output = {
            "reasoning_content_base64": base64.b64encode(reasoning_bytes).decode("ascii"),
            "content_base64": base64.b64encode(content_bytes).decode("ascii"),
        }
    return {
        "index": result.index,
        "output_sha256": result.output_sha256,
        "reasoning_sha256": text_sha256(result.reasoning_content),
        "content_sha256": text_sha256(result.content),
        "reasoning_utf8_bytes": len(reasoning_bytes),
        "content_utf8_bytes": len(content_bytes),
        "completion_tokens": result.completion_tokens,
        "finish_reason": result.finish_reason,
        "exact_output": exact_output,
    }


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
        server_performance=None,
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
    expected_system_fingerprint: str | None = None,
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
        performance_records = 0
        system_fingerprint_records = 0
        server_performance: dict[str, Any] | None = None
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
                    system_fingerprint = value.get("system_fingerprint")
                    if system_fingerprint is not None:
                        if not isinstance(system_fingerprint, str):
                            raise BenchmarkError(
                                "stream system_fingerprint must be a string or null"
                            )
                        system_fingerprint_records += 1
                        if (
                            expected_system_fingerprint is not None
                            and system_fingerprint != expected_system_fingerprint
                        ):
                            raise BenchmarkError(
                                "stream system_fingerprint disagrees with the "
                                "vLLM runtime manifest"
                            )
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
                    metadata = value.get("metadata")
                    if isinstance(metadata, dict) and "performance" in metadata:
                        performance_records += 1
                        if performance_records != 1:
                            raise BenchmarkError(
                                "stream emitted multiple performance records"
                            )
                        server_performance = validate_chat_performance_metadata(
                            metadata["performance"],
                            "stream metadata.performance",
                        )
            for data in parser.finish():
                if data == "[DONE]":
                    done = True
                else:
                    raise BenchmarkError("stream ended with an unterminated non-DONE SSE event")
        if not done:
            raise BenchmarkError("stream ended without [DONE]")
        if usage_records != 1:
            raise BenchmarkError("stream did not emit exactly one usage record")
        if (
            expected_system_fingerprint is not None
            and system_fingerprint_records == 0
        ):
            raise BenchmarkError(
                "stream did not report the vLLM runtime manifest system_fingerprint"
            )
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
            server_performance=server_performance,
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


def validate_batching_engine_snapshot(snapshot: Any) -> dict[str, Any]:
    if not isinstance(snapshot, dict):
        raise BenchmarkError("diagnostics omit decode_runtime.batching_engine")
    for field in ("max_decode_batch", "max_observed_batch_size", *COUNTER_FIELDS):
        value = snapshot.get(field)
        if isinstance(value, bool) or not isinstance(value, (int, float)) or value < 0:
            raise BenchmarkError(f"invalid batching diagnostic {field}={value!r}")
    for field in ("actor_cycle_idle_ms", "actor_cycle_idle_count"):
        _nonnegative_int(snapshot.get(field), f"batching diagnostic {field}")
    for field in ("total_actor_cycle_idle_ms", "max_actor_cycle_idle_ms"):
        _nonnegative_number(snapshot.get(field), f"batching diagnostic {field}")
    if snapshot.get("actor_cycle_idle_source") not in {
        "default",
        "config_file",
        "environment",
    }:
        raise BenchmarkError("invalid batching diagnostic actor_cycle_idle_source")
    if not isinstance(snapshot.get("actor_cycle_idle_active"), bool):
        raise BenchmarkError("invalid batching diagnostic actor_cycle_idle_active")
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
    for field in ("actor_cycle_idle_ms", "actor_cycle_idle_source"):
        if after[field] != before[field]:
            raise BenchmarkError(f"batching {field} changed during run")
    if after["actor_cycle_idle_count"] < before["actor_cycle_idle_count"]:
        raise BenchmarkError("batching actor-cycle idle count regressed")
    if after["total_actor_cycle_idle_ms"] < before["total_actor_cycle_idle_ms"]:
        raise BenchmarkError("batching actor-cycle idle wall time regressed")
    if after["max_actor_cycle_idle_ms"] < before["max_actor_cycle_idle_ms"]:
        raise BenchmarkError("batching actor-cycle idle maximum regressed")
    result.update(
        {
            "actor_cycle_idle_ms": int(after["actor_cycle_idle_ms"]),
            "actor_cycle_idle_source": after["actor_cycle_idle_source"],
            "actor_cycle_idle_active_end": after["actor_cycle_idle_active"],
            "actor_cycle_idle_count": (
                int(after["actor_cycle_idle_count"])
                - int(before["actor_cycle_idle_count"])
            ),
            "actor_cycle_idle_seconds": (
                after["total_actor_cycle_idle_ms"]
                - before["total_actor_cycle_idle_ms"]
            )
            / 1_000.0,
            "process_max_actor_cycle_idle_ms": after["max_actor_cycle_idle_ms"],
        }
    )
    forwards = result["total_decode_forwards"]
    result["mean_decode_rows_per_forward"] = (
        result["total_decode_rows"] / forwards if forwards else 0.0
    )
    result["batched_decode_forward_fraction"] = (
        result["total_batched_decode_forwards"] / forwards if forwards else 0.0
    )
    return result


def validate_request_snapshot(value: Any) -> dict[str, int]:
    snapshot = _object(value, "diagnostics.requests")
    required = {*REQUEST_COUNTER_FIELDS, "active", "active_peak"}
    _exact_keys(snapshot, required, "diagnostics.requests")
    normalized = {
        field: _nonnegative_int(snapshot[field], f"diagnostics.requests.{field}")
        for field in required
    }
    status_total = sum(normalized[field] for field in REQUEST_COUNTER_FIELDS[1:])
    if normalized["total"] != status_total:
        raise BenchmarkError(
            "diagnostics.requests.total disagrees with ok/error/timeout/rejected"
        )
    if normalized["active"] > normalized["active_peak"]:
        raise BenchmarkError("diagnostics.requests.active exceeds active_peak")
    return normalized


def validate_rocm_graph_snapshot(value: Any) -> dict[str, Any]:
    label = "diagnostics.decode_runtime.rocm_graphs"
    snapshot = _object(value, label)
    base = {
        field: snapshot.get(field)
        for field in (
            "state",
            "unavailable_reason",
            "requested",
            "capture_requested",
            "enabled",
            "capture_enabled",
            *ROCM_GRAPH_COUNTER_FIELDS,
            *ROCM_GRAPH_GAUGE_FIELDS,
            "fallbacks",
        )
    }
    normalized = validate_rocm_graph_record(
        base,
        label,
        gauge_suffixes=("",),
        fallback_max_field="max_duration_micros",
        fallback_reason_fields=ROCM_GRAPH_FALLBACK_REASON_FIELDS,
    )
    available = normalized["state"] in {"enabled", "disabled"}
    parity_fields = (
        *ROCM_GRAPH_BATCHED_CAPTURE_COUNTER_FIELDS,
        *ROCM_GRAPH_CAPTURE_PARITY_COUNTER_FIELDS,
    )
    for field in parity_fields:
        item = snapshot.get(field)
        if available:
            normalized[field] = _nonnegative_int(item, f"{label}.{field}")
        else:
            if item is not None:
                raise BenchmarkError(f"{label}.{field} must be null when unavailable")
            normalized[field] = None
    if available:
        if normalized["batched_capture_attempts"] != (
            normalized["batched_capture_successes"]
            + normalized["batched_capture_deferrals"]
            + normalized["batched_capture_failures"]
        ):
            raise BenchmarkError(f"{label} batched capture outcomes disagree")
        if normalized["capture_parity_checks"] != (
            normalized["capture_parity_passes"]
            + normalized["capture_parity_failures"]
            + normalized["capture_parity_errors"]
        ):
            raise BenchmarkError(f"{label} capture parity outcomes disagree")
        if (
            normalized["batched_capture_successes"]
            > normalized["capture_parity_passes"]
        ):
            raise BenchmarkError(
                f"{label} successful batched captures lack parity admission"
            )
        if (
            normalized["capture_parity_checks"] > 0
            and normalized["capture_parity_compared_bytes"] == 0
        ):
            raise BenchmarkError(f"{label} parity checks cover zero bytes")
    return normalized


def server_diagnostics_snapshot(health: dict[str, Any]) -> dict[str, Any]:
    requests = validate_request_snapshot(health.get("requests"))
    runtime = _object(health.get("decode_runtime"), "diagnostics.decode_runtime")
    batching_value = runtime.get("batching_engine")
    if batching_value is None:
        raise BenchmarkError("diagnostics.decode_runtime.batching_engine is unavailable")
    batching_engine = validate_batching_engine_snapshot(batching_value)
    rocm_graphs = validate_rocm_graph_snapshot(runtime.get("rocm_graphs"))
    return {
        "request_route": "batching_engine",
        "requests": requests,
        "batching_engine": batching_engine,
        "rocm_graphs": rocm_graphs,
    }


def settled_server_diagnostics_snapshot(
    url: str, headers: dict[str, str], timeout_secs: float
) -> dict[str, Any]:
    """Read one actor-idle boundary outside the measured request window."""
    deadline: float | None = None
    while True:
        snapshot = server_diagnostics_snapshot(fetch_json(url, headers, timeout_secs))
        batching = snapshot["batching_engine"]
        if batching is None or not batching["actor_cycle_idle_active"]:
            return snapshot
        if deadline is None:
            configured_seconds = batching["actor_cycle_idle_ms"] / 1_000.0
            deadline = time.monotonic() + min(
                timeout_secs, configured_seconds + 1.0
            )
        if time.monotonic() >= deadline:
            return snapshot
        time.sleep(0.005)


def request_delta(before: dict[str, int], after: dict[str, int]) -> dict[str, int]:
    result: dict[str, int] = {}
    for field in REQUEST_COUNTER_FIELDS:
        if after[field] < before[field]:
            raise BenchmarkError(
                f"request counter {field} regressed from {before[field]} to {after[field]}"
            )
        result[field] = after[field] - before[field]
    if after["active_peak"] < before["active_peak"]:
        raise BenchmarkError(
            "request active_peak regressed from "
            f"{before['active_peak']} to {after['active_peak']}"
        )
    result["active_end"] = after["active"]
    result["process_active_peak"] = after["active_peak"]
    return result


def rocm_graph_delta(before: dict[str, Any], after: dict[str, Any]) -> dict[str, Any]:
    before_available = before["state"] in {"enabled", "disabled"}
    after_available = after["state"] in {"enabled", "disabled"}
    if before_available != after_available:
        raise BenchmarkError("ROCm graph diagnostic availability changed during run")
    if not before_available:
        if before != after:
            raise BenchmarkError("unavailable ROCm graph diagnostics changed during run")
        return {
            "state": after["state"],
            "unavailable_reason": after["unavailable_reason"],
            "requested": None,
            "capture_requested": None,
            "enabled": None,
            "capture_enabled": None,
            **{field: None for field in ROCM_GRAPH_COUNTER_FIELDS},
            **{
                f"{field}_{boundary}": None
                for field in ROCM_GRAPH_GAUGE_FIELDS
                for boundary in ("start", "end")
            },
            "fallbacks": None,
            "capture_parity": None,
        }

    for field in ("requested", "capture_requested"):
        if before[field] != after[field]:
            raise BenchmarkError(f"ROCm graph {field} changed during run")
    for field in ("enabled", "capture_enabled"):
        if not before[field] and after[field]:
            raise BenchmarkError(f"ROCm graph {field} rearmed during run")

    result: dict[str, Any] = {
        "state": after["state"],
        "unavailable_reason": after["unavailable_reason"],
        "requested": after["requested"],
        "capture_requested": after["capture_requested"],
        "enabled": after["enabled"],
        "capture_enabled": after["capture_enabled"],
    }
    for field in ROCM_GRAPH_COUNTER_FIELDS:
        if after[field] < before[field]:
            raise BenchmarkError(
                f"ROCm graph counter {field} regressed from "
                f"{before[field]} to {after[field]}"
            )
        result[field] = after[field] - before[field]
    for field in ROCM_GRAPH_GAUGE_FIELDS:
        result[f"{field}_start"] = before[field]
        result[f"{field}_end"] = after[field]

    before_fallbacks = before["fallbacks"]
    after_fallbacks = after["fallbacks"]
    fallback_delta: dict[str, int] = {}
    for field in ROCM_GRAPH_FALLBACK_COUNTER_FIELDS:
        if after_fallbacks[field] < before_fallbacks[field]:
            raise BenchmarkError(
                f"ROCm graph fallback counter {field} regressed from "
                f"{before_fallbacks[field]} to {after_fallbacks[field]}"
            )
        fallback_delta[field] = after_fallbacks[field] - before_fallbacks[field]
    if (
        after_fallbacks["max_duration_micros"]
        < before_fallbacks["max_duration_micros"]
    ):
        raise BenchmarkError("ROCm graph fallback max duration regressed")
    fallback_delta["process_max_duration_micros"] = after_fallbacks[
        "max_duration_micros"
    ]
    result["fallbacks"] = fallback_delta
    parity_delta: dict[str, int] = {}
    for field in (
        *ROCM_GRAPH_BATCHED_CAPTURE_COUNTER_FIELDS,
        *ROCM_GRAPH_CAPTURE_PARITY_COUNTER_FIELDS,
    ):
        if after[field] < before[field]:
            raise BenchmarkError(
                f"ROCm graph parity counter {field} regressed from "
                f"{before[field]} to {after[field]}"
            )
        parity_delta[field] = after[field] - before[field]
    for field in ROCM_GRAPH_CAPTURE_PARITY_BOUNDARY_FIELDS:
        parity_delta[f"{field}_start"] = before[field]
        parity_delta[f"{field}_end"] = after[field]
    result["capture_parity"] = parity_delta
    return result


def server_diagnostics_delta(
    before: dict[str, Any], after: dict[str, Any]
) -> dict[str, Any]:
    if before["request_route"] != after["request_route"]:
        raise BenchmarkError("effective request route changed during run")
    before_batching = before["batching_engine"]
    after_batching = after["batching_engine"]
    return {
        "schema": SERVER_DIAGNOSTICS_SCHEMA,
        "request_route": after["request_route"],
        "requests": request_delta(before["requests"], after["requests"]),
        "batching_engine": batching_delta(before_batching, after_batching),
        "rocm_graphs": rocm_graph_delta(
            before["rocm_graphs"], after["rocm_graphs"]
        ),
    }


def server_diagnostics_has_no_errors(server: dict[str, Any]) -> bool:
    requests = server["requests"]
    if requests["error"] or requests["timeout"] or requests["rejected"]:
        return False
    batching_engine = server["batching_engine"]
    if batching_engine is not None and batching_engine["total_errors"]:
        return False
    graph = server.get("rocm_graphs")
    return graph is None or graph["failures"] in {None, 0}


def server_actor_cycle_idle_accounted(server: dict[str, Any]) -> bool:
    batching = server["batching_engine"]
    if batching is None:
        return True
    if batching["actor_cycle_idle_active_end"]:
        return False
    configured = batching["actor_cycle_idle_ms"]
    count = batching["actor_cycle_idle_count"]
    elapsed = batching["actor_cycle_idle_seconds"]
    maximum = batching["process_max_actor_cycle_idle_ms"]
    if configured == 0:
        return count == 0 and elapsed == 0 and maximum == 0
    return (
        batching["actor_cycle_idle_source"] in {"config_file", "environment"}
        and count > 0
        and elapsed > 0
        and maximum > 0
    )


def server_rocm_graph_execution_accounted(server: dict[str, Any]) -> bool:
    graph = server["rocm_graphs"]
    if graph["state"] in {"busy", "unavailable"}:
        return graph["unavailable_reason"] == "backend_without_graph_runner"
    if not graph["capture_requested"]:
        return graph["failures"] == 0 and graph["fallbacks"]["total"] == 0
    return (
        graph["capture_enabled"]
        and graph["failures"] == 0
        and graph["fallbacks"]["total"] == 0
        and graph["capture_successes"] + graph["replay_successes"] > 0
    )


def server_rocm_graph_capture_parity_accounted(server: dict[str, Any]) -> bool:
    graph = server["rocm_graphs"]
    parity = graph.get("capture_parity")
    if graph["state"] in {"busy", "unavailable"}:
        return parity is None
    if parity is None:
        return False
    batching = server["batching_engine"]
    measured_multi_row = (
        batching is not None and batching["total_batched_decode_forwards"] > 0
    )
    if not graph["capture_requested"] or not measured_multi_row:
        return (
            parity["batched_capture_attempts"] == 0
            and parity["capture_parity_checks"] == 0
        )
    return (
        graph["capture_enabled"]
        and graph["failures"] == 0
        and parity["capture_parity_failures_end"] == 0
        and parity["capture_parity_errors_end"] == 0
        and parity["capture_parity_checks_end"]
        == parity["capture_parity_passes_end"]
        and parity["batched_capture_successes_end"]
        == parity["capture_parity_passes_end"]
        and parity["capture_parity_passes_end"] > 0
        and parity["batched_capture_successes"] == parity["capture_parity_passes"]
        and (
            parity["capture_parity_checks"] == 0
            or parity["capture_parity_compared_bytes"] > 0
        )
    )


def server_request_accounting_matches(
    server: dict[str, Any], concurrency: int
) -> bool:
    requests = server["requests"]
    return (
        requests["total"] == concurrency
        and requests["ok"] == concurrency
        and requests["active_end"] == 0
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
    output_evidence_mode: str = "hashes",
    engine_name: str | None = None,
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
        if server.get("schema") in SERVER_DIAGNOSTICS_SCHEMAS:
            request_errors = {
                field: server["requests"][field]
                for field in ("error", "timeout", "rejected")
            }
            batching_errors = (
                server["batching_engine"]["total_errors"]
                if server["batching_engine"] is not None
                else 0
            )
            direct_errors = (
                server["decode_batcher"]["failed_jobs"]
                if server.get("decode_batcher") is not None
                else 0
            )
            graph = server.get("rocm_graphs")
            graph_failures = graph["failures"] if graph is not None else None
            gates.extend(
                [
                    gate(
                        "server_reported_no_errors",
                        server_diagnostics_has_no_errors(server),
                        (
                            f"route={server['request_route']}; request={request_errors}; "
                            f"batching_engine={batching_errors}; "
                            f"decode_batcher={direct_errors}; "
                            f"rocm_graph_failures={graph_failures}"
                        ),
                    ),
                    gate(
                        "server_request_accounting",
                        server_request_accounting_matches(server, concurrency),
                        (
                            f"total={server['requests']['total']}, "
                            f"ok={server['requests']['ok']}, "
                            f"active_end={server['requests']['active_end']}; "
                            f"expected={concurrency}"
                        ),
                    ),
                ]
            )
            if server["schema"] == SERVER_DIAGNOSTICS_SCHEMA:
                graph = server["rocm_graphs"]
                fallback_count = (
                    None
                    if graph["fallbacks"] is None
                    else graph["fallbacks"]["total"]
                )
                multi_row_fallback_count = (
                    None
                    if graph["fallbacks"] is None
                    else graph["fallbacks"]["multi_row_batch_unsupported"]
                )
                gates.append(
                    gate(
                        "rocm_graph_execution_accounted",
                        server_rocm_graph_execution_accounted(server),
                        (
                            f"state={graph['state']}; "
                            f"capture_requested={graph['capture_requested']}; "
                            f"capture_successes={graph['capture_successes']}; "
                            f"replay_successes={graph['replay_successes']}; "
                            f"failures={graph['failures']}; "
                            f"fallbacks={fallback_count}; "
                            "multi_row_batch_unsupported="
                            f"{multi_row_fallback_count}"
                        ),
                    )
                )
                parity = graph["capture_parity"]
                gates.append(
                    gate(
                        "rocm_graph_capture_parity_accounted",
                        server_rocm_graph_capture_parity_accounted(server),
                        (
                            "not available for this backend"
                            if parity is None
                            else (
                                f"batched_successes={parity['batched_capture_successes']}; "
                                f"checks={parity['capture_parity_checks']}; "
                                f"passes={parity['capture_parity_passes']}; "
                                f"failures={parity['capture_parity_failures']}; "
                                f"errors={parity['capture_parity_errors']}; "
                                f"passes_end={parity['capture_parity_passes_end']}; "
                                f"compared_bytes={parity['capture_parity_compared_bytes']}; "
                                f"duration_micros={parity['capture_parity_duration_micros']}"
                            )
                        ),
                    )
                )
                batching = server["batching_engine"]
                gates.append(
                    gate(
                        "actor_cycle_idle_accounted",
                        server_actor_cycle_idle_accounted(server),
                        (
                            "not applicable: batching actor inactive"
                            if batching is None
                            else (
                                f"configured_ms={batching['actor_cycle_idle_ms']}; "
                                f"source={batching['actor_cycle_idle_source']}; "
                                f"active_end={batching['actor_cycle_idle_active_end']}; "
                                f"count={batching['actor_cycle_idle_count']}; "
                                f"seconds={batching['actor_cycle_idle_seconds']:.6f}; "
                                f"process_max_ms={batching['process_max_actor_cycle_idle_ms']:.3f}"
                            )
                        ),
                    )
                )
        else:
            gates.append(
                gate(
                    "server_reported_no_errors",
                    server["total_errors"] == 0,
                    f"batching-engine error delta: {server['total_errors']}",
                )
            )
    request_performance_rows: list[dict[str, Any]] | None = None
    request_phase_summary: dict[str, Any] | None = None
    if engine_name == "kiln":
        request_performance_rows = [
            {"index": result.index, "performance": result.server_performance}
            for result in sorted(successes, key=lambda result: result.index)
            if result.server_performance is not None
        ]
        request_phase_summary = build_request_phase_summary(
            row["performance"] for row in request_performance_rows
        )
        latency_count = request_phase_summary["latency_request_count"]
        complete = (
            len(request_performance_rows) == len(successes)
            and latency_count == len(successes)
        )
        gates.append(
            gate(
                "request_performance_accounted",
                complete,
                (
                    f"performance={len(request_performance_rows)}/{len(successes)} "
                    f"successful requests; latency={latency_count}/{len(successes)}"
                ),
            )
        )
    output_evidence_rows = [
        output_evidence(result, output_evidence_mode)
        for result in sorted(successes, key=lambda result: result.index)
    ]
    output_rows = [
        output_set_evidence_row(evidence) for evidence in output_evidence_rows
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
        "output_evidence": output_evidence_rows,
        "request_performance": request_performance_rows,
        "request_phase_summary": request_phase_summary,
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
    expected_system_fingerprint: str | None = None,
) -> dict[str, Any]:
    bodies: list[dict[str, Any]] = []
    prompts: set[str] = set()
    for index in range(concurrency):
        prompt = deterministic_prompt(
            args.prompt_set_id,
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
            diagnostics_before = settled_server_diagnostics_snapshot(
                diagnostics_url, headers, args.timeout_secs
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
            expected_system_fingerprint=expected_system_fingerprint,
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
            diagnostics_after = settled_server_diagnostics_snapshot(
                diagnostics_url, headers, args.timeout_secs
            )
            assert diagnostics_before is not None
            server_delta = server_diagnostics_delta(
                diagnostics_before, diagnostics_after
            )
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
        output_evidence_mode=args.output_evidence,
        engine_name=args.engine,
    )


def eager_reference_execution_summary(reference: dict[str, Any]) -> dict[str, Any]:
    rows = list(reference.get("runs", []))
    if reference.get("warmup") is not None:
        rows.insert(0, reference["warmup"])
    observed = all(
        isinstance(row.get("server"), dict)
        and row["server"].get("schema") == SERVER_DIAGNOSTICS_SCHEMA
        for row in rows
    )
    graphs = [row["server"]["rocm_graphs"] for row in rows] if observed else []
    capture_successes = sum(graph["capture_successes"] or 0 for graph in graphs)
    replay_successes = sum(graph["replay_successes"] or 0 for graph in graphs)
    failures = sum(graph["failures"] or 0 for graph in graphs)
    fallbacks = sum(
        (graph["fallbacks"] or {}).get("total", 0) for graph in graphs
    )
    return {
        "row_count": len(rows),
        "all_rows_observed": observed,
        "all_rows_capture_disabled": observed
        and all(graph["capture_requested"] is False for graph in graphs),
        "capture_successes": capture_successes,
        "replay_successes": replay_successes,
        "failures": failures,
        "fallbacks": fallbacks,
    }


def compare_reference(receipt: dict[str, Any], reference_path: Path) -> dict[str, Any]:
    try:
        reference_bytes = reference_path.read_bytes()
        reference = strict_json_loads(reference_bytes)
    except Exception as exc:
        raise BenchmarkError(
            f"cannot load reference receipt {reference_path}: {type(exc).__name__}: {exc}"
        ) from exc
    validate_benchmark_receipt(reference)
    if reference.get("driver_version") not in REFERENCE_COMPATIBLE_DRIVER_VERSIONS:
        raise BenchmarkError(
            "reference receipt must use a comparison-compatible driver version in "
            f"{sorted(REFERENCE_COMPATIBLE_DRIVER_VERSIONS)}"
        )
    if reference.get("workload_fingerprint") != receipt.get("workload_fingerprint"):
        raise BenchmarkError("reference receipt has a different workload fingerprint")
    if receipt.get("driver_version") in TYPED_MEMORY_SOURCE_DRIVER_VERSIONS:
        current_memory = receipt["memory_sampler"]
        reference_memory = reference["memory_sampler"]
        if (
            current_memory["source"] != reference_memory["source"]
            or current_memory["interval_ms"] != reference_memory["interval_ms"]
        ):
            raise BenchmarkError(
                "reference receipt uses different device-memory telemetry"
            )
        if current_memory["source"] == "drm_vram_used":
            same_device = current_memory["path"] == reference_memory["path"]
        else:
            if (
                reference.get("driver_version")
                not in TYPED_MEMORY_SOURCE_DRIVER_VERSIONS
            ):
                raise BenchmarkError(
                    "NVML comparison requires a typed-memory driver reference receipt"
                )
            same_device = (
                current_memory["device"]["uuid"]
                == reference_memory["device"]["uuid"]
                and current_memory["device"]["total_bytes"]
                == reference_memory["device"]["total_bytes"]
            )
        if not same_device:
            raise BenchmarkError(
                "reference receipt measured a different accelerator device"
            )
    current_model = receipt.get("engine", {}).get("model_identity", {})
    reference_model = reference.get("engine", {}).get("model_identity", {})
    if current_model.get("content_sha256") != reference_model.get("content_sha256"):
        raise BenchmarkError("reference receipt has different model content")
    reference_role = receipt["reference_role"]
    verdict_effect = "required" if reference_role == "qualification_gate" else "evidence_only"
    reference_execution: dict[str, Any] | None = None
    if reference_role == "same_artifact_graph_eager_discriminator":
        if reference.get("driver_version") != DRIVER_VERSION:
            raise BenchmarkError(
                "same-artifact graph/eager discrimination requires a current-driver reference"
            )
        if reference.get("verdict") != "passed":
            raise BenchmarkError(
                "same-artifact graph/eager discrimination requires a passed eager reference"
            )
        if reference.get("reference_role") != "qualification_gate":
            raise BenchmarkError(
                "same-artifact graph/eager discrimination requires an ordinary eager reference"
            )
        current_engine = receipt["engine"]
        reference_engine = reference["engine"]
        if current_engine["name"] != "kiln" or reference_engine["name"] != "kiln":
            raise BenchmarkError(
                "same-artifact graph/eager discrimination requires two Kiln receipts"
            )
        if (
            current_engine["runtime_artifact"]["sha256"]
            != reference_engine["runtime_artifact"]["sha256"]
            or current_engine["runtime_identity"] != reference_engine["runtime_identity"]
        ):
            raise BenchmarkError(
                "same-artifact graph/eager discrimination requires the identical runtime artifact"
            )
        reference_execution = eager_reference_execution_summary(reference)
        if not (
            reference_execution["row_count"] > 0
            and reference_execution["all_rows_observed"]
            and reference_execution["all_rows_capture_disabled"]
            and reference_execution["capture_successes"] == 0
            and reference_execution["replay_successes"] == 0
            and reference_execution["failures"] == 0
            and reference_execution["fallbacks"] == 0
        ):
            raise BenchmarkError(
                "same-artifact graph/eager reference does not prove graph-disabled execution"
            )
        candidate_rows = receipt.get("runs", [])
        if not candidate_rows or not all(
            isinstance(row.get("server"), dict)
            and row["server"].get("schema") == SERVER_DIAGNOSTICS_SCHEMA
            and row["server"]["rocm_graphs"]["capture_requested"] is True
            and server_rocm_graph_execution_accounted(row["server"])
            and server_rocm_graph_capture_parity_accounted(row["server"])
            for row in candidate_rows
        ):
            raise BenchmarkError(
                "same-artifact graph/eager candidate lacks measured graph parity evidence"
            )
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
            mismatches.append(output_mismatch_detail(current, expected))
    return {
        "reference_receipt_sha256": "sha256:" + hashlib.sha256(reference_bytes).hexdigest(),
        "reference_engine": reference.get("engine"),
        "reference_role": reference_role,
        "verdict_effect": verdict_effect,
        "reference_execution": reference_execution,
        "comparison_mode": comparison_mode,
        "matched": not mismatches,
        "mismatches": mismatches,
    }


def first_divergent_utf8_byte(left: bytes, right: bytes) -> int | None:
    for index, (left_byte, right_byte) in enumerate(zip(left, right)):
        if left_byte != right_byte:
            return index
    if len(left) != len(right):
        return min(len(left), len(right))
    return None


def exact_output_bytes(evidence: dict[str, Any]) -> tuple[bytes, bytes] | None:
    exact = evidence["exact_output"]
    if exact is None:
        return None
    return (
        base64.b64decode(exact["reasoning_content_base64"], validate=True),
        base64.b64decode(exact["content_base64"], validate=True),
    )


def output_mismatch_detail(
    current: dict[str, Any], expected: dict[str, Any]
) -> dict[str, Any]:
    current_evidence = {row["index"]: row for row in current["output_evidence"]}
    expected_evidence = {row["index"]: row for row in expected["output_evidence"]}
    request_mismatches: list[dict[str, Any]] = []
    compared_fields = (
        "output_sha256",
        "reasoning_sha256",
        "content_sha256",
        "reasoning_utf8_bytes",
        "content_utf8_bytes",
        "completion_tokens",
        "finish_reason",
    )
    for index in sorted(set(current_evidence) | set(expected_evidence)):
        actual = current_evidence.get(index)
        reference = expected_evidence.get(index)
        if actual is None or reference is None:
            raise BenchmarkError(
                "output evidence does not cover the same successful request indices"
            )
        fields = sorted(
            name for name in compared_fields if actual[name] != reference[name]
        )
        if not fields:
            continue
        actual_exact = exact_output_bytes(actual)
        reference_exact = exact_output_bytes(reference)
        exact_compared = actual_exact is not None and reference_exact is not None
        reasoning_offset = content_offset = None
        if exact_compared:
            assert actual_exact is not None and reference_exact is not None
            reasoning_offset = first_divergent_utf8_byte(
                reference_exact[0], actual_exact[0]
            )
            content_offset = first_divergent_utf8_byte(
                reference_exact[1], actual_exact[1]
            )
        request_mismatches.append(
            {
                "index": index,
                "fields": fields,
                "expected_output_sha256": reference["output_sha256"],
                "actual_output_sha256": actual["output_sha256"],
                "exact_output_compared": exact_compared,
                "reasoning_first_divergent_utf8_byte": reasoning_offset,
                "content_first_divergent_utf8_byte": content_offset,
            }
        )
    if not request_mismatches:
        raise BenchmarkError(
            "output_set_sha256 differs but per-request output evidence matches"
        )
    indices = [row["index"] for row in request_mismatches]
    return {
        "concurrency": current["concurrency"],
        "repeat": current["repeat"],
        "reason": "output_mismatch",
        "mismatch_count": len(request_mismatches),
        "mismatched_request_indices": indices,
        "request_mismatches": request_mismatches,
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


def wait_for_owned_server_models(
    server: OwnedServer,
    base_url: str,
    headers: dict[str, str],
) -> list[str]:
    wall_deadline = time.monotonic() + server.config.startup_timeout_seconds
    last_error = "server has not accepted a readiness probe"
    while True:
        returncode = server.process.poll()
        if returncode is not None:
            raise BenchmarkError(
                f"owned server exited during startup with status {returncode}:\n"
                f"{server_log_tail(server.log_path)}"
            )
        now = time.monotonic()
        if now >= wall_deadline:
            raise BenchmarkError(
                f"owned server did not become ready within "
                f"{server.config.startup_timeout_seconds:.3f} wall seconds; last probe: "
                f"{last_error}\n{server_log_tail(server.log_path)}"
            )
        remaining = wall_deadline - now
        try:
            return probe_models(base_url, headers, min(2.0, remaining))
        except BenchmarkError as exc:
            last_error = str(exc)
        time.sleep(min(server.config.readiness_poll_interval_seconds, remaining))


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
        "prompt_set_id": args.prompt_set_id,
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


def workload_fingerprint(
    workload: dict[str, Any], *, driver_version: str = DRIVER_VERSION
) -> str:
    if driver_version not in PROMPT_SET_IDENTITY_DRIVER_VERSIONS:
        return canonical_sha256(workload)
    comparison_workload = dict(workload)
    comparison_workload.pop("run_id")
    return canonical_sha256(comparison_workload)


def print_run(row: dict[str, Any]) -> None:
    server = row.get("server") or {}
    route = "unobserved"
    if server.get("schema") in SERVER_DIAGNOSTICS_SCHEMAS:
        route = server["request_route"]
        route_diagnostics = (
            server["batching_engine"]
            if route == "batching_engine"
            else server.get("decode_batcher")
        ) or {}
        width = route_diagnostics.get("process_max_observed_batch")
        mean = (
            route_diagnostics.get("mean_decode_rows_per_forward")
            if route == "batching_engine"
            else route_diagnostics.get("mean_runner_calls_per_executed_row")
        )
    else:
        route = "batching_engine" if server else route
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
        f"route={route} route_max={width_text} route_mean={mean_text} "
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
        help="Unique operational identity for this receipt and owned server log",
    )
    parser.add_argument(
        "--prompt-set-id",
        help="Stable model-visible prompt identity shared by comparable runs",
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
    parser.add_argument(
        "--memory-source",
        choices=("auto", "drm", "macos", "nvml"),
        default="auto",
        help=(
            "whole-device memory telemetry source; auto requires exactly one "
            "unambiguous DRM, macOS unified-memory, or NVML device"
        ),
    )
    parser.add_argument("--memory-path", default="auto")
    parser.add_argument(
        "--memory-device-index",
        type=int,
        help="physical NVML device index; index or UUID is required on multi-GPU hosts",
    )
    parser.add_argument(
        "--memory-device-uuid",
        help="stable NVML GPU UUID; preferred when CUDA device indices may be remapped",
    )
    parser.add_argument("--memory-sample-ms", type=int, default=50)
    parser.add_argument(
        "--model-fingerprint-read-mib-per-second",
        type=int,
        default=DEFAULT_MODEL_FINGERPRINT_READ_MIB_PER_SECOND,
        help=(
            "optional cumulative read-rate limit across both model-integrity "
            "passes; zero disables it (default: unlimited)"
        ),
    )
    parser.add_argument("--require-memory", action="store_true")
    parser.add_argument("--memory-limit-bytes", type=int)
    server_ownership = parser.add_mutually_exclusive_group()
    server_ownership.add_argument(
        "--server-pid",
        type=int,
        help="local process-group leader for an already-running server",
    )
    server_ownership.add_argument(
        "--server-launch-config",
        type=Path,
        help="typed argv-only server lifecycle owned by this benchmark",
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
        "--reference-role",
        choices=tuple(sorted(REFERENCE_ROLES)),
        default="qualification_gate",
        help=(
            "make cross-process comparison verdict-gating, or retain it as "
            "reproducibility evidence for an exact same-artifact eager/graph "
            "discriminator whose in-process graph parity is separately gated"
        ),
    )
    parser.add_argument(
        "--output-evidence",
        choices=("hashes", "full"),
        default="hashes",
        help=(
            "retain per-request hashes, or bounded base64 output text for "
            "first-divergence diagnostics"
        ),
    )
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
    if args.run_id is not None and not re.fullmatch(
        r"[A-Za-z0-9][A-Za-z0-9._-]{2,127}", args.run_id
    ):
        parser.error("run-id must be 3..128 portable identifier characters")
    if args.prompt_set_id is not None and not re.fullmatch(
        r"[A-Za-z0-9][A-Za-z0-9._-]{2,127}", args.prompt_set_id
    ):
        parser.error("prompt-set-id must be 3..128 portable identifier characters")
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
    if args.memory_device_index is not None and args.memory_device_index < 0:
        parser.error("memory-device-index must be non-negative")
    if args.memory_device_uuid is not None and re.fullmatch(
        r"GPU-[A-Za-z0-9-]{8,120}", args.memory_device_uuid
    ) is None:
        parser.error("memory-device-uuid must be a complete NVML GPU UUID")
    if args.memory_device_index is not None and args.memory_device_uuid is not None:
        parser.error("memory-device-index and memory-device-uuid are mutually exclusive")
    if args.memory_source == "auto":
        has_nvml_selector = (
            args.memory_device_index is not None
            or args.memory_device_uuid is not None
        )
        if args.memory_path != "auto" and has_nvml_selector:
            parser.error(
                "memory-path and an NVML device selector cannot both select a device"
            )
        if args.memory_path != "auto":
            args.memory_source = "drm"
        elif has_nvml_selector:
            args.memory_source = "nvml"
    elif args.memory_source == "drm" and (
        args.memory_device_index is not None or args.memory_device_uuid is not None
    ):
        parser.error("NVML device selectors cannot be combined with memory-source drm")
    elif args.memory_source == "nvml" and args.memory_path != "auto":
        parser.error("memory-path cannot be combined with memory-source nvml")
    elif args.memory_source == "macos" and (
        args.memory_path != "auto"
        or args.memory_device_index is not None
        or args.memory_device_uuid is not None
    ):
        parser.error(
            "memory-source macos cannot be combined with DRM or NVML selectors"
        )
    if args.model_fingerprint_read_mib_per_second != 0 and not (
        MIN_MODEL_FINGERPRINT_READ_MIB_PER_SECOND
        <= args.model_fingerprint_read_mib_per_second
        <= MAX_MODEL_FINGERPRINT_READ_MIB_PER_SECOND
    ):
        parser.error(
            "model-fingerprint-read-mib-per-second must be zero or in "
            f"{MIN_MODEL_FINGERPRINT_READ_MIB_PER_SECOND}..="
            f"{MAX_MODEL_FINGERPRINT_READ_MIB_PER_SECOND}"
        )
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
    if (
        args.reference_role == "same_artifact_graph_eager_discriminator"
        and args.reference_receipt is None
    ):
        parser.error(
            "same_artifact_graph_eager_discriminator requires --reference-receipt"
        )
    if (
        args.reference_role == "same_artifact_graph_eager_discriminator"
        and args.engine != "kiln"
    ):
        parser.error(
            "same_artifact_graph_eager_discriminator is available only for Kiln"
        )
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    owned_server: OwnedServer | None = None
    owned_shutdown: dict[str, Any] | None = None
    owned_log: dict[str, Any] | None = None
    attached_process: AttachedProcessGroup | None = None
    try:
        if args.validate_receipt is not None:
            if (
                args.out is not None
                or args.reference_receipt is not None
                or args.server_pid is not None
                or args.server_launch_config is not None
            ):
                raise BenchmarkError(
                    "--validate-receipt cannot be combined with output, reference, "
                    "or server runtime arguments"
                )
            for path in args.validate_receipt:
                validate_benchmark_receipt_path(path)
                print(f"OK {path}")
            return 0
        if args.run_id is None or args.prompt_set_id is None:
            raise BenchmarkError(
                "measured runs require explicit --run-id and --prompt-set-id values"
            )
        if args.run_id == args.prompt_set_id:
            raise BenchmarkError("--run-id and --prompt-set-id must be distinct")
        if args.server_pid is None and args.server_launch_config is None:
            raise BenchmarkError(
                "measured runs require exactly one server owner"
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
        launch_config: ServerLaunchConfig | None = None
        if args.server_launch_config is not None:
            launch_config = load_server_launch_config(args.server_launch_config)
            require_owned_base_url_unbound(args.base_url)
            if args.engine == "kiln":
                if Path(runtime_artifact["path"]).resolve() != Path(
                    launch_config.command[0]
                ):
                    raise BenchmarkError(
                        "owned Kiln launch executable must equal --runtime-artifact"
                    )
            else:
                validate_vllm_owned_launch(launch_config, runtime_manifest)
        try:
            initial_model_identity = fingerprint_model(
                args.model_path,
                args.model,
                max_read_mib_per_second=(
                    args.model_fingerprint_read_mib_per_second or None
                ),
            )
            model_identity = bind_model_identity(initial_model_identity)
        except ModelFingerprintError as exc:
            raise BenchmarkError(f"model fingerprint failed: {exc}") from exc
        if args.server_launch_config is not None:
            assert launch_config is not None
            owned_server = launch_owned_server(launch_config, args.run_id)
        else:
            assert args.server_pid is not None
            attached_process = AttachedProcessGroup.attach(args.server_pid)

        headers = {
            "Accept": "text/event-stream",
            "Content-Type": "application/json",
            "User-Agent": f"kiln-serving-benchmark/{DRIVER_VERSION}",
        }
        if args.api_key:
            headers["Authorization"] = f"Bearer {args.api_key}"
        models: list[str] = []
        health_version = None
        runtime_execution_identity: dict[str, Any] | None = None
        server_startup_error: Exception | None = None
        try:
            models = (
                wait_for_owned_server_models(
                    owned_server,
                    args.base_url,
                    headers,
                )
                if owned_server is not None
                else probe_models(args.base_url, headers, args.timeout_secs)
            )
            if owned_server is not None:
                verify_owned_listener(owned_server, args.base_url)
            if args.model not in models:
                raise BenchmarkError(
                    f"requested model {args.model!r} is absent from /v1/models: {models}"
                )
            if args.engine == "kiln":
                health = fetch_json(
                    f"{args.base_url}/health", headers, args.timeout_secs
                )
                health_version = health.get("version")
                runtime_execution_identity = _object(
                    health.get("execution_identity"),
                    "Kiln health.execution_identity",
                )
                if (
                    runtime_execution_identity.get("executable_sha256")
                    != runtime_artifact["sha256"]
                ):
                    raise BenchmarkError(
                        "Kiln health execution identity does not match --runtime-artifact"
                    )
        except Exception as exc:
            server_startup_error = exc

        diagnostics_url: str | None
        if args.diagnostics_url == "none":
            diagnostics_url = None
        elif args.diagnostics_url == "auto":
            diagnostics_url = f"{args.base_url}/health" if args.engine == "kiln" else None
        else:
            diagnostics_url = args.diagnostics_url

        try:
            memory_counter = resolve_memory_counter(
                source=args.memory_source,
                drm_path=args.memory_path,
                nvml_device_index=args.memory_device_index,
                nvml_device_uuid=args.memory_device_uuid,
            )
        except DeviceMemoryError as exc:
            raise BenchmarkError(f"device-memory telemetry is unavailable: {exc}") from exc
        workload = workload_contract(args, sizes)
        sampler = MemorySampler(memory_counter, args.memory_sample_ms)
        memory_sampler_identity = sampler.receipt_identity()
        memory_device = memory_sampler_identity["device"]
        if (
            memory_device is not None
            and args.memory_limit_bytes > memory_device["total_bytes"]
        ):
            sampler.stop()
            raise BenchmarkError(
                "memory-limit-bytes exceeds the selected NVML device capacity"
            )
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

        try:
            if server_startup_error is not None:
                record_completion_failure("server_startup", server_startup_error)
            else:
                try:
                    sampler.start()
                    if args.warmup_requests:
                        warmup = run_once(
                            args=args,
                            concurrency=args.warmup_requests,
                            repeat=-1,
                            max_tokens=min(16, args.max_tokens),
                            phase=f"warmup-c{args.warmup_requests:03d}",
                            headers=headers,
                            sampler=sampler,
                            diagnostics_url=diagnostics_url,
                            expected_system_fingerprint=(
                                None
                                if runtime_manifest is None
                                else runtime_manifest["system_fingerprint"]
                            ),
                        )
                        print(
                            f"[warmup] {warmup['verdict']} "
                            f"ok={warmup['success_count']}/{warmup['request_count']}"
                        )

                    if warmup is None or warmup["verdict"] == "passed":
                        for concurrency in sizes:
                            for repeat in range(args.repeats):
                                row = run_once(
                                    args=args,
                                    concurrency=concurrency,
                                    repeat=repeat,
                                    max_tokens=args.max_tokens,
                                    phase=f"measure-c{concurrency:03d}-r{repeat:03d}",
                                    headers=headers,
                                    sampler=sampler,
                                    diagnostics_url=diagnostics_url,
                                    expected_system_fingerprint=(
                                        None
                                        if runtime_manifest is None
                                        else runtime_manifest["system_fingerprint"]
                                    ),
                                )
                                runs.append(row)
                                print_run(row)
                    elif warmup is not None:
                        record_completion_failure("warmup", "warmup verdict failed")
                except Exception as exc:
                    record_completion_failure("measurement", exc)
        finally:
            try:
                sampler.stop()
            except Exception as exc:
                record_completion_failure("memory_sampler_stop", exc)

        if args.engine == "kiln":
            def verify_execution_identity() -> None:
                if runtime_execution_identity is None:
                    raise BenchmarkError(
                        "Kiln execution identity is unavailable because startup did not complete"
                    )
                health_after = fetch_json(
                    f"{args.base_url}/health", headers, args.timeout_secs
                )
                if health_after.get("execution_identity") != runtime_execution_identity:
                    raise BenchmarkError("Kiln execution identity changed during measurement")

            run_finalization_check("execution_identity_unchanged", verify_execution_identity)
        else:
            finalization_checks["execution_identity_unchanged"] = "not_applicable"

        if owned_server is not None:

            def verify_owned_server_shutdown() -> None:
                nonlocal owned_shutdown, owned_log
                owned_shutdown, owned_log, failures = finalize_owned_server(owned_server)
                if failures:
                    raise BenchmarkError(
                        "owned server finalization failed: "
                        + "; ".join(
                            f"{type(exc).__name__}: {exc}" for exc in failures
                        )
                    )
                assert owned_shutdown is not None
                assert owned_log is not None
                if owned_shutdown["forced"]:
                    raise BenchmarkError("owned server required SIGKILL during shutdown")
                if owned_shutdown["process_group_alive_end"]:
                    raise BenchmarkError("owned server process group survived shutdown")
                if (
                    owned_shutdown["returncode"]
                    not in owned_shutdown["acceptable_exit_codes"]
                ):
                    raise BenchmarkError(
                        "owned server returned an unacceptable exit status "
                        f"{owned_shutdown['returncode']}"
                    )

            run_finalization_check("server_shutdown", verify_owned_server_shutdown)
        else:
            finalization_checks["server_shutdown"] = "not_applicable"

        def verify_model_identity() -> None:
            try:
                model_after_raw = fingerprint_model(
                    args.model_path,
                    args.model,
                    max_read_mib_per_second=(
                        args.model_fingerprint_read_mib_per_second or None
                    ),
                )
                model_after = bind_model_identity(model_after_raw)
            except ModelFingerprintError as exc:
                raise BenchmarkError(f"model fingerprint recheck failed: {exc}") from exc
            if model_after != model_identity:
                raise BenchmarkError("model identity changed during measurement")

        run_finalization_check("model_identity_unchanged", verify_model_identity)

        run_finalization_check(
            "repository_unchanged",
            lambda: require_repository_unchanged(repo),
        )

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

        if owned_server is not None:
            if owned_shutdown is None or owned_log is None:
                shutdown_failures = [
                    failure["detail"]
                    for failure in completion_failures
                    if failure["phase"] == "server_shutdown"
                ]
                detail = (
                    shutdown_failures[-1]
                    if shutdown_failures
                    else "server shutdown evidence is unavailable"
                )
                raise BenchmarkError(
                    "cannot serialize owned server lifecycle evidence: " + detail
                )
            server_lifecycle = {
                "mode": "owned_process_group",
                "launch_config": owned_server.config.record,
                "log": owned_log,
                "shutdown": owned_shutdown,
            }
        elif attached_process is not None:
            server_lifecycle = {
                "mode": "attached_process_group",
                "launch_config": None,
                "log": None,
                "shutdown": None,
            }
        else:
            server_lifecycle = {
                "mode": "not_configured",
                "launch_config": None,
                "log": None,
                "shutdown": None,
            }

        receipt: dict[str, Any] = {
            "schema": SCHEMA,
            "driver_version": DRIVER_VERSION,
            "reference_role": args.reference_role,
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
            "workload_fingerprint": workload_fingerprint(workload),
            "memory_sampler": {
                **memory_sampler_identity,
                "interval_ms": args.memory_sample_ms,
            },
            "diagnostics": {
                "url": diagnostics_url,
                "timed_request_path_affected": False,
            },
            "server_lifecycle": server_lifecycle,
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
            and all(
                status in {"passed", "not_applicable"}
                for status in finalization_checks.values()
            )
            and (warmup is None or warmup["verdict"] == "passed")
            and len(runs) == len(sizes) * args.repeats
            and all(row["verdict"] == "passed" for row in runs)
            and (
                receipt.get("comparison", {}).get("matched", True)
                or args.reference_role
                == "same_artifact_graph_eager_discriminator"
            )
        )
        receipt["verdict"] = "passed" if passed else "failed"
        receipt["receipt_sha256"] = canonical_sha256(receipt)
        validate_benchmark_receipt(receipt)
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
        if owned_server is not None and owned_server.process.poll() is None:
            try:
                owned_shutdown = shutdown_owned_server(owned_server)
            except Exception as exc:
                print(f"owned server cleanup error: {exc}", file=sys.stderr)
        if owned_server is not None and not owned_server.log_handle.closed:
            try:
                close_owned_server_log(owned_server)
            except Exception as exc:
                print(f"owned server log cleanup error: {exc}", file=sys.stderr)


if __name__ == "__main__":
    raise SystemExit(main())
