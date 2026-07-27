#!/usr/bin/env python3
"""Capture a repeatable vLLM runtime manifest from an owned-launch document."""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import importlib.util
import json
import math
import os
import platform
import re
import resource
import signal
import stat
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Sequence


ROOT = Path(__file__).resolve().parents[2]
BENCHMARK_SCRIPT = ROOT / "scripts" / "bench-concurrent-batch.py"
BENCHMARK_SPEC = importlib.util.spec_from_file_location(
    "capture_vllm_benchmark_contract",
    BENCHMARK_SCRIPT,
)
assert BENCHMARK_SPEC is not None and BENCHMARK_SPEC.loader is not None
benchmark = importlib.util.module_from_spec(BENCHMARK_SPEC)
sys.modules[BENCHMARK_SPEC.name] = benchmark
BENCHMARK_SPEC.loader.exec_module(benchmark)


MAX_MANIFEST_BYTES = 1024 * 1024
MAX_STDERR_BYTES = 8 * 1024 * 1024
DEFAULT_TIMEOUT_SECONDS = 1800.0
CAPTURE_TERMINATION_GRACE_SECONDS = 30.0
WSL_THERMAL_EXEC = ROOT / "scripts" / "qualification" / "wsl_thermal_exec.py"
WSL_SCOPE_EXEC = ROOT / "scripts" / "qualification" / "wsl_scope_exec.py"
WSL_SCOPE_PAYLOAD = ROOT / "scripts" / "qualification" / "wsl_scope_payload.py"
LINUX_NAMESPACE_EXEC = ROOT / "scripts" / "qualification" / "linux_namespace_exec.py"
WSL_PLATFORM = ROOT / "scripts" / "qualification" / "wsl_platform.py"
UNSHARE_EXECUTABLE = Path("/usr/bin/unshare")
WSL_SCOPE_CPU_POLL_INTERVAL_MS = 5
WSL_SCOPE_UNPACED_CPU_QUOTA_PERCENT = 0
WSL_SCOPE_UNPACED_MEMORY_MAX_BYTES = 0
WSL_THERMAL_EVENT_PREFIX = "wsl2-thermal: "
WSL_THERMAL_EVENT_SCHEMA = "kiln.wsl2-thermal-event.v1"
WSL_SCOPE_EVENT_PREFIX = "wsl2-scope: "
WSL_SCOPE_EVENT_SCHEMA_UNPACED = "kiln.wsl2-scope-event.v1"
WSL_SCOPE_EVENT_SCHEMA_PACED = "kiln.wsl2-scope-event.v2"
WSL_SCOPE_EVENT_SCHEMA = WSL_SCOPE_EVENT_SCHEMA_PACED
WSL_THERMAL_PREFLIGHT_KEYS = {
    "schema",
    "event",
    "policy_id",
    "policy_sha256",
    "host_millicelsius",
    "gpu_millicelsius",
    "host_limit_millicelsius",
    "gpu_limit_millicelsius",
}
WSL_THERMAL_COMPLETE_KEYS = {
    "schema",
    "event",
    "policy_id",
    "policy_sha256",
    "supervision_outcome",
    "failure_reason",
    "child_returncode",
    "sample_count",
    "starting_host_millicelsius",
    "starting_gpu_millicelsius",
    "peak_host_millicelsius",
    "peak_gpu_millicelsius",
    "ending_host_millicelsius",
    "ending_gpu_millicelsius",
    "safe_handoff_stable_samples",
}
WSL_SCOPE_START_KEYS = {
    "schema",
    "event",
    "unit",
    "cgroup",
    "containment",
    "memory_max_bytes",
    "memory_swap_max_bytes",
    "pids_max",
    "cpu_quota_percent",
    "cpu_controller",
    "cpu_poll_interval_ms",
    "runtime_max_seconds",
    "thermal_policy_sha256",
    "thermal_pacing",
}
WSL_SCOPE_START_KEYS_UNPACED = WSL_SCOPE_START_KEYS - {"thermal_pacing"}
WSL_SCOPE_COMPLETE_KEYS = {
    "schema",
    "event",
    "unit",
    "duration_seconds",
    "cpu_usage_usec",
    "cpu_allowed_usec",
    "cpu_quota_percent",
    "memory_peak_bytes",
    "memory_events",
    "pids_peak",
    "scope_removed",
    "child_returncode",
    "reason",
    "thermal_pacing",
}
WSL_SCOPE_COMPLETE_KEYS_UNPACED = WSL_SCOPE_COMPLETE_KEYS - {"thermal_pacing"}
WSL_SCOPE_PACING_START_KEYS = {
    "policy_sha256",
    "mode",
    "telemetry_source",
    "freeze_verification",
    "host_start_millicelsius",
    "host_resume_millicelsius",
    "gpu_start_millicelsius",
    "gpu_resume_millicelsius",
    "resume_stable_samples",
    "timeout_seconds",
}
WSL_SCOPE_PACING_COMPLETE_KEYS = {
    "policy_sha256",
    "mode",
    "active",
    "sample_count",
    "pause_count",
    "completed_pause_count",
    "total_pause_seconds",
    "longest_pause_seconds",
    "peak_host_millicelsius",
    "peak_gpu_millicelsius",
    "ending_host_millicelsius",
    "ending_gpu_millicelsius",
}


class CaptureError(RuntimeError):
    """Raised when a runtime manifest cannot be captured reproducibly."""


@dataclasses.dataclass(frozen=True)
class Capture:
    payload: bytes
    manifest: dict[str, Any]
    stderr_bytes: int
    stderr_sha256: str
    wsl2_thermal: dict[str, Any] | None
    wsl2_scope: dict[str, Any] | None


@dataclasses.dataclass(frozen=True)
class Wsl2ThermalSupervision:
    path: Path
    repository_path: str
    policy: Any
    unshare_path: Path


@dataclasses.dataclass(frozen=True)
class Wsl2ScopeSupervision:
    unshare_path: Path
    thermal: Wsl2ThermalSupervision | None


def require_clean_repository(root: Path = ROOT) -> str:
    """Require one clean committed source before runtime identity capture."""

    completed = subprocess.run(
        ["git", "status", "--porcelain=v1", "--untracked-files=normal"],
        cwd=root,
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    if completed.returncode != 0:
        details = completed.stderr.decode("utf-8", errors="replace").strip()
        raise CaptureError(f"cannot inspect repository state: {details}")
    if completed.stdout:
        raise CaptureError("repository must be clean before vLLM manifest capture")
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=root,
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    ).stdout.strip()
    if len(commit) != 40:
        raise CaptureError("repository HEAD is not a full commit hash")
    return commit


def manifest_command(config: Any) -> list[str]:
    """Insert manifest-only mode without duplicating the checked launch argv."""

    benchmark.validate_vllm_owned_launch(config)
    command = list(config.command)
    try:
        boundary = command.index("--", 2)
    except ValueError as exc:
        raise CaptureError("owned vLLM launch has no explicit argument boundary") from exc
    command.insert(boundary, "--manifest-only")
    return command


def _repository_regular_file(path: Path, label: str) -> tuple[Path, str]:
    absolute = Path(os.path.abspath(os.fspath(path if path.is_absolute() else ROOT / path)))
    try:
        repository_path = absolute.relative_to(ROOT).as_posix()
    except ValueError as exc:
        raise CaptureError(f"{label} must stay inside {ROOT}") from exc
    current = ROOT
    for part in Path(repository_path).parts:
        current = current / part
        try:
            info = current.lstat()
        except OSError as exc:
            raise CaptureError(f"cannot inspect {label} {current}: {exc}") from exc
        if stat.S_ISLNK(info.st_mode):
            raise CaptureError(f"{label} must not use symlinks: {current}")
    try:
        final_info = absolute.lstat()
    except OSError as exc:
        raise CaptureError(f"cannot inspect {label} {absolute}: {exc}") from exc
    if not stat.S_ISREG(final_info.st_mode):
        raise CaptureError(f"{label} must be a regular file: {absolute}")
    tracked = subprocess.run(
        ["git", "ls-files", "--error-unmatch", "--", repository_path],
        cwd=ROOT,
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    if tracked.returncode != 0:
        raise CaptureError(f"{label} must be tracked by the current repository")
    head_blob = subprocess.run(
        ["git", "rev-parse", f"HEAD:{repository_path}"],
        cwd=ROOT,
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    working_blob = subprocess.run(
        ["git", "hash-object", "--no-filters", "--", os.fspath(absolute)],
        cwd=ROOT,
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    if head_blob.returncode != 0 or working_blob.returncode != 0:
        raise CaptureError(f"cannot bind {label} to the current repository commit")
    if head_blob.stdout.strip() != working_blob.stdout.strip():
        raise CaptureError(f"{label} bytes do not match the current repository commit")
    return absolute, repository_path


def _load_wsl2_scope_prerequisites() -> Path:
    _repository_regular_file(WSL_SCOPE_EXEC, "WSL2 scope supervisor")
    _repository_regular_file(WSL_SCOPE_PAYLOAD, "WSL2 scope payload")
    _repository_regular_file(LINUX_NAMESPACE_EXEC, "Linux namespace helper")
    _repository_regular_file(WSL_PLATFORM, "WSL2 platform helper")
    unshare_path = UNSHARE_EXECUTABLE
    try:
        unshare_info = unshare_path.lstat()
    except OSError as exc:
        raise CaptureError(f"cannot inspect util-linux unshare: {exc}") from exc
    if (
        not stat.S_ISREG(unshare_info.st_mode)
        or not os.access(unshare_path, os.X_OK)
        or unshare_info.st_uid != 0
        or unshare_info.st_mode & 0o022
        or unshare_path.name != "unshare"
    ):
        raise CaptureError(
            "WSL2 manifest capture requires root-owned non-writable /usr/bin/unshare"
        )
    return unshare_path


def load_wsl2_thermal_supervision(path: Path) -> Wsl2ThermalSupervision:
    absolute, repository_path = _repository_regular_file(
        path,
        "WSL2 thermal policy",
    )
    _repository_regular_file(WSL_THERMAL_EXEC, "WSL2 thermal supervisor")
    unshare_path = _load_wsl2_scope_prerequisites()
    try:
        policy = benchmark.wsl_thermal_exec.load_policy(absolute)
    except benchmark.wsl_thermal_exec.ThermalGuardError as exc:
        raise CaptureError(f"invalid WSL2 thermal policy: {exc}") from exc
    if policy.pacing is None:
        raise CaptureError(
            "WSL2 manifest capture requires a v2 cgroup thermal pacing policy"
        )
    return Wsl2ThermalSupervision(
        path=absolute,
        repository_path=repository_path,
        policy=policy,
        unshare_path=unshare_path,
    )


def load_platform_supervision(
    path: Path | None,
) -> Wsl2ScopeSupervision | None:
    running_on_wsl2 = "microsoft-standard-wsl2" in platform.release().lower()
    if not running_on_wsl2 and path is not None:
        raise CaptureError("--wsl2-thermal-policy is only valid on WSL2")
    if not running_on_wsl2:
        return None
    if path is None:
        return Wsl2ScopeSupervision(
            unshare_path=_load_wsl2_scope_prerequisites(),
            thermal=None,
        )
    thermal = load_wsl2_thermal_supervision(path)
    return Wsl2ScopeSupervision(
        unshare_path=thermal.unshare_path,
        thermal=thermal,
    )


def supervised_manifest_command(
    config: Any,
    supervision: Wsl2ScopeSupervision | None,
    timeout_seconds: float,
) -> list[str]:
    command = manifest_command(config)
    if supervision is None:
        return command
    thermal = supervision.thermal
    memory_max_bytes = (
        WSL_SCOPE_UNPACED_MEMORY_MAX_BYTES
        if thermal is None
        else benchmark.WSL2_SCOPE_MEMORY_MAX_BYTES
    )
    handoff_seconds = (
        0.0 if thermal is None else thermal.policy.handoff_timeout_seconds
    )
    scope_runtime_seconds = (
        timeout_seconds - handoff_seconds - CAPTURE_TERMINATION_GRACE_SECONDS
    )
    if scope_runtime_seconds < 1.0:
        raise CaptureError(
            "capture timeout cannot contain the WSL2 scope"
            + (" plus thermal handoff" if thermal is not None else "")
        )
    namespaced_command = [
        os.fspath(supervision.unshare_path),
        "--user",
        "--map-root-user",
        "--net",
        "--pid",
        "--fork",
        "--kill-child=SIGKILL",
        "--mount",
        "--mount-proc=/proc",
        sys.executable,
        os.fspath(LINUX_NAMESPACE_EXEC),
        "--",
        *command,
    ]
    scoped_command = [
        sys.executable,
        os.fspath(WSL_SCOPE_EXEC),
        "--memory-max-bytes",
        str(memory_max_bytes),
        "--pids-max",
        str(benchmark.WSL2_SCOPE_PIDS_MAX),
        "--cpu-quota-percent",
        str(
            WSL_SCOPE_UNPACED_CPU_QUOTA_PERCENT
            if thermal is None
            else benchmark.WSL2_SCOPE_CPU_QUOTA_PERCENT
        ),
        "--cpu-poll-interval-ms",
        str(WSL_SCOPE_CPU_POLL_INTERVAL_MS),
        "--runtime-max-seconds",
        f"{scope_runtime_seconds:g}",
    ]
    if thermal is not None:
        scoped_command.extend(
            [
                "--thermal-pacing-policy",
                os.fspath(thermal.path),
            ]
        )
    scoped_command.extend(["--", *namespaced_command])
    if thermal is None:
        return scoped_command
    return [
        sys.executable,
        os.fspath(WSL_THERMAL_EXEC),
        "--policy",
        os.fspath(thermal.path),
        "--",
        *scoped_command,
    ]


def _limit_capture_files() -> None:
    resource.setrlimit(resource.RLIMIT_FSIZE, (MAX_STDERR_BYTES, MAX_STDERR_BYTES))


def _terminate_capture_session(
    process: subprocess.Popen[Any],
    grace_seconds: float,
) -> None:
    if process.poll() is not None:
        return
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        process.wait()
        return
    try:
        process.wait(timeout=grace_seconds)
        return
    except subprocess.TimeoutExpired:
        pass
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    process.wait()


def _run_capture_child(
    command: Sequence[str],
    *,
    working_directory: Path,
    stdout: Any,
    stderr: Any,
    timeout_seconds: float,
    termination_grace_seconds: float,
    environment: dict[str, str] | None = None,
) -> int:
    process = subprocess.Popen(
        list(command),
        cwd=working_directory,
        env=environment,
        stdin=subprocess.DEVNULL,
        stdout=stdout,
        stderr=stderr,
        close_fds=True,
        preexec_fn=_limit_capture_files,
        start_new_session=True,
    )
    try:
        return process.wait(timeout=timeout_seconds)
    except subprocess.TimeoutExpired as exc:
        _terminate_capture_session(process, termination_grace_seconds)
        raise CaptureError(
            f"vLLM runtime manifest capture exceeded {timeout_seconds} seconds"
        ) from exc
    except BaseException:
        _terminate_capture_session(process, termination_grace_seconds)
        raise


def _thermal_integer(value: Any, label: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise CaptureError(f"{label} must be an integer at or above {minimum}")
    return value


def validate_wsl2_thermal_stderr(
    stderr_payload: bytes,
    supervision: Wsl2ThermalSupervision,
) -> dict[str, Any]:
    events: list[dict[str, Any]] = []
    for raw_line in stderr_payload.decode("utf-8", errors="replace").splitlines():
        if not raw_line.startswith(WSL_THERMAL_EVENT_PREFIX):
            continue
        payload = raw_line.removeprefix(WSL_THERMAL_EVENT_PREFIX).encode("utf-8")
        try:
            event = benchmark.strict_json_loads(payload)
        except Exception as exc:
            raise CaptureError(f"WSL2 thermal event is not strict JSON: {exc}") from exc
        if not isinstance(event, dict):
            raise CaptureError("WSL2 thermal event must be an object")
        events.append(event)
    if [event.get("event") for event in events] != ["preflight", "complete"]:
        raise CaptureError(
            "WSL2 thermal supervision must emit exactly preflight then complete"
        )

    preflight, complete = events
    if set(preflight) != WSL_THERMAL_PREFLIGHT_KEYS:
        raise CaptureError("WSL2 preflight event fields are invalid")
    if set(complete) != WSL_THERMAL_COMPLETE_KEYS:
        raise CaptureError("WSL2 complete event fields are invalid")
    policy = supervision.policy
    for index, event in enumerate(events):
        label = ("preflight", "complete")[index]
        if event.get("schema") != WSL_THERMAL_EVENT_SCHEMA:
            raise CaptureError(f"WSL2 {label} event schema is invalid")
        if (
            event.get("policy_id") != policy.policy_id
            or event.get("policy_sha256") != policy.content_sha256
        ):
            raise CaptureError(f"WSL2 {label} event policy identity is invalid")
    if (
        preflight.get("host_limit_millicelsius")
        != policy.host_limit_millicelsius
        or preflight.get("gpu_limit_millicelsius")
        != policy.gpu_limit_millicelsius
    ):
        raise CaptureError("WSL2 preflight hard limits do not match the policy")
    if complete.get("supervision_outcome") != "child_exit":
        raise CaptureError("WSL2 thermal supervision did not report child_exit")
    if complete.get("failure_reason") is not None:
        raise CaptureError("WSL2 thermal supervision reported a failure")
    if complete.get("child_returncode") != 0:
        raise CaptureError("WSL2 thermal supervision child did not exit zero")

    values = {
        field: _thermal_integer(
            complete.get(field),
            f"WSL2 complete {field}",
            minimum=1,
        )
        for field in (
            "sample_count",
            "starting_host_millicelsius",
            "starting_gpu_millicelsius",
            "peak_host_millicelsius",
            "peak_gpu_millicelsius",
            "ending_host_millicelsius",
            "ending_gpu_millicelsius",
            "safe_handoff_stable_samples",
        )
    }
    if (
        values["starting_host_millicelsius"]
        != preflight.get("host_millicelsius")
        or values["starting_gpu_millicelsius"]
        != preflight.get("gpu_millicelsius")
    ):
        raise CaptureError("WSL2 preflight and complete starting samples disagree")
    if (
        values["peak_host_millicelsius"] >= policy.host_limit_millicelsius
        or values["peak_gpu_millicelsius"] >= policy.gpu_limit_millicelsius
    ):
        raise CaptureError("WSL2 thermal peak reached a hard limit")
    if (
        values["ending_host_millicelsius"] > policy.handoff_host_millicelsius
        or values["ending_gpu_millicelsius"] > policy.handoff_gpu_millicelsius
    ):
        raise CaptureError("WSL2 thermal safe handoff targets were not reached")
    if (
        values["safe_handoff_stable_samples"] != policy.handoff_stable_samples
        or values["sample_count"] < policy.handoff_stable_samples + 1
    ):
        raise CaptureError("WSL2 thermal stable-handoff sample evidence is invalid")
    for sensor in ("host", "gpu"):
        starting = values[f"starting_{sensor}_millicelsius"]
        peak = values[f"peak_{sensor}_millicelsius"]
        ending = values[f"ending_{sensor}_millicelsius"]
        if peak < max(starting, ending):
            raise CaptureError(f"WSL2 {sensor} peak is below an endpoint")
    return {
        "mechanism": "per-capture-windows-thermal-zone-nvml-v1",
        "policy_path": supervision.repository_path,
        "policy_id": policy.policy_id,
        "policy_sha256": policy.content_sha256,
        **values,
    }


def _scope_number(value: Any, label: str, *, minimum: float = 0) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
        or value < minimum
    ):
        raise CaptureError(f"{label} must be finite and at or above {minimum}")
    return float(value)


def _validate_scope_thermal_pacing(
    start_value: Any,
    complete_value: Any,
    supervision: Wsl2ThermalSupervision,
) -> dict[str, Any]:
    if (
        not isinstance(start_value, dict)
        or set(start_value) != WSL_SCOPE_PACING_START_KEYS
    ):
        raise CaptureError("WSL2 scope thermal pacing start fields are invalid")
    if (
        not isinstance(complete_value, dict)
        or set(complete_value) != WSL_SCOPE_PACING_COMPLETE_KEYS
    ):
        raise CaptureError("WSL2 scope thermal pacing complete fields are invalid")
    policy = supervision.policy
    pacing = policy.pacing
    if pacing is None:
        raise CaptureError("WSL2 scope thermal pacing policy is unavailable")
    expected_start = {
        "policy_sha256": policy.content_sha256,
        "mode": pacing.mode,
        "telemetry_source": "outer-supervisor-inherited-pipe-v1",
        "freeze_verification": "cgroup-freeze-and-events-roundtrip-v1",
        "host_start_millicelsius": pacing.host_start_millicelsius,
        "host_resume_millicelsius": pacing.host_resume_millicelsius,
        "gpu_start_millicelsius": pacing.gpu_start_millicelsius,
        "gpu_resume_millicelsius": pacing.gpu_resume_millicelsius,
        "resume_stable_samples": pacing.resume_stable_samples,
        "timeout_seconds": pacing.timeout_seconds,
    }
    if start_value != expected_start:
        raise CaptureError("WSL2 scope thermal pacing start does not match the policy")
    if (
        complete_value.get("policy_sha256") != policy.content_sha256
        or complete_value.get("mode") != pacing.mode
        or complete_value.get("active") is not False
    ):
        raise CaptureError("WSL2 scope thermal pacing did not finish inactive")
    integer_values = {
        field: _thermal_integer(
            complete_value.get(field),
            f"WSL2 scope thermal pacing {field}",
            minimum=minimum,
        )
        for field, minimum in (
            ("sample_count", 1),
            ("pause_count", 0),
            ("completed_pause_count", 0),
            ("peak_host_millicelsius", 1),
            ("peak_gpu_millicelsius", 1),
            ("ending_host_millicelsius", 1),
            ("ending_gpu_millicelsius", 1),
        )
    }
    if integer_values["pause_count"] != integer_values["completed_pause_count"]:
        raise CaptureError("WSL2 scope thermal pacing has an incomplete pause")
    total_pause = _scope_number(
        complete_value.get("total_pause_seconds"),
        "WSL2 scope thermal pacing total pause",
    )
    longest_pause = _scope_number(
        complete_value.get("longest_pause_seconds"),
        "WSL2 scope thermal pacing longest pause",
    )
    if longest_pause > total_pause:
        raise CaptureError("WSL2 scope thermal pacing pause durations are invalid")
    if integer_values["pause_count"] == 0 and (
        total_pause != 0 or longest_pause != 0
    ):
        raise CaptureError("WSL2 scope thermal pacing reported time without a pause")
    if integer_values["pause_count"] > 0 and (
        longest_pause <= 0 or longest_pause >= pacing.timeout_seconds
    ):
        raise CaptureError("WSL2 scope thermal pacing pause duration is invalid")
    if (
        integer_values["peak_host_millicelsius"]
        >= policy.host_limit_millicelsius
        or integer_values["peak_gpu_millicelsius"]
        >= policy.gpu_limit_millicelsius
    ):
        raise CaptureError("WSL2 scope thermal pacing peak reached a hard limit")
    for sensor in ("host", "gpu"):
        if (
            integer_values[f"peak_{sensor}_millicelsius"]
            < integer_values[f"ending_{sensor}_millicelsius"]
        ):
            raise CaptureError(
                f"WSL2 scope thermal pacing {sensor} peak is below its ending sample"
            )
    return {
        **complete_value,
        "total_pause_seconds": total_pause,
        "longest_pause_seconds": longest_pause,
    }


def validate_wsl2_scope_stderr(
    stderr_payload: bytes,
    supervision: Wsl2ScopeSupervision,
    expected_runtime_max_seconds: float,
) -> dict[str, Any]:
    events: list[dict[str, Any]] = []
    for raw_line in stderr_payload.decode("utf-8", errors="replace").splitlines():
        if not raw_line.startswith(WSL_SCOPE_EVENT_PREFIX):
            continue
        payload = raw_line.removeprefix(WSL_SCOPE_EVENT_PREFIX).encode("utf-8")
        try:
            event = benchmark.strict_json_loads(payload)
        except Exception as exc:
            raise CaptureError(f"WSL2 scope event is not strict JSON: {exc}") from exc
        if not isinstance(event, dict):
            raise CaptureError("WSL2 scope event must be an object")
        events.append(event)
    if [event.get("event") for event in events] != ["start", "complete"]:
        raise CaptureError("WSL2 scope must emit exactly start then complete")
    start, complete = events
    thermal = supervision.thermal
    expected_start_keys = (
        WSL_SCOPE_START_KEYS_UNPACED
        if thermal is None
        else WSL_SCOPE_START_KEYS
    )
    expected_complete_keys = (
        WSL_SCOPE_COMPLETE_KEYS_UNPACED
        if thermal is None
        else WSL_SCOPE_COMPLETE_KEYS
    )
    expected_schema = (
        WSL_SCOPE_EVENT_SCHEMA_UNPACED
        if thermal is None
        else WSL_SCOPE_EVENT_SCHEMA_PACED
    )
    if set(start) != expected_start_keys:
        raise CaptureError("WSL2 scope start event fields are invalid")
    if set(complete) != expected_complete_keys:
        raise CaptureError("WSL2 scope complete event fields are invalid")
    for label, event in (("start", start), ("complete", complete)):
        if event.get("schema") != expected_schema:
            raise CaptureError(f"WSL2 scope {label} event schema is invalid")

    unit = start.get("unit")
    if (
        not isinstance(unit, str)
        or not re.fullmatch(r"kiln-wsl-scope-[0-9a-f]{32}", unit)
        or complete.get("unit") != unit
    ):
        raise CaptureError("WSL2 scope unit identity is invalid")
    expected_cgroup = (
        f"/sys/fs/cgroup/user.slice/user-{os.getuid()}.slice/"
        f"user@{os.getuid()}.service/app.slice/{unit}.scope"
    )
    if start.get("cgroup") != expected_cgroup:
        raise CaptureError("WSL2 scope cgroup path is invalid")
    cpu_quota_percent = (
        WSL_SCOPE_UNPACED_CPU_QUOTA_PERCENT
        if thermal is None
        else benchmark.WSL2_SCOPE_CPU_QUOTA_PERCENT
    )
    memory_max_bytes = (
        WSL_SCOPE_UNPACED_MEMORY_MAX_BYTES
        if thermal is None
        else benchmark.WSL2_SCOPE_MEMORY_MAX_BYTES
    )
    expected_start = {
        "containment": benchmark.WSL2_NETWORK_BOUNDARY,
        "memory_max_bytes": memory_max_bytes,
        "memory_swap_max_bytes": 0,
        "pids_max": benchmark.WSL2_SCOPE_PIDS_MAX,
        "cpu_quota_percent": cpu_quota_percent,
        "cpu_controller": (
            "not_configured"
            if thermal is None
            else "usage-feedback-cgroup-freeze-v1"
        ),
        "cpu_poll_interval_ms": WSL_SCOPE_CPU_POLL_INTERVAL_MS,
        "thermal_policy_sha256": (
            None if thermal is None else thermal.policy.content_sha256
        ),
    }
    for field, expected in expected_start.items():
        if start.get(field) != expected:
            raise CaptureError(f"WSL2 scope start {field} is invalid")
    pacing_evidence = (
        None
        if thermal is None
        else _validate_scope_thermal_pacing(
            start.get("thermal_pacing"),
            complete.get("thermal_pacing"),
            thermal,
        )
    )
    runtime_max_seconds = _scope_number(
        start.get("runtime_max_seconds"),
        "WSL2 scope runtime maximum",
        minimum=1,
    )
    if runtime_max_seconds != expected_runtime_max_seconds:
        raise CaptureError("WSL2 scope runtime maximum is invalid")
    duration_seconds = _scope_number(
        complete.get("duration_seconds"),
        "WSL2 scope duration",
        minimum=0,
    )
    if duration_seconds > runtime_max_seconds + 1.0:
        raise CaptureError("WSL2 scope duration exceeded its bounded allowance")
    if (
        complete.get("cpu_quota_percent") != cpu_quota_percent
        or complete.get("scope_removed") is not True
        or complete.get("child_returncode") != 0
        or complete.get("reason") is not None
    ):
        raise CaptureError("WSL2 scope did not complete its required lifecycle")
    cpu_usage = _thermal_integer(
        complete.get("cpu_usage_usec"),
        "WSL2 scope CPU usage",
    )
    if thermal is None:
        if complete.get("cpu_allowed_usec") is not None:
            raise CaptureError(
                "WSL2 unpaced scope must not report a CPU allowance"
            )
        cpu_allowed: int | None = None
    else:
        cpu_allowed = _thermal_integer(
            complete.get("cpu_allowed_usec"),
            "WSL2 scope CPU allowance",
        )
        expected_allowed = int(
            duration_seconds
            * 1_000_000
            * benchmark.WSL2_SCOPE_CPU_QUOTA_PERCENT
            / 100
        )
        if cpu_allowed != expected_allowed or cpu_usage > cpu_allowed:
            raise CaptureError("WSL2 scope CPU accounting is invalid")
    memory_peak = _thermal_integer(
        complete.get("memory_peak_bytes"),
        "WSL2 scope memory peak",
        minimum=1,
    )
    pids_peak = _thermal_integer(
        complete.get("pids_peak"),
        "WSL2 scope PID peak",
        minimum=1,
    )
    if memory_max_bytes > 0 and memory_peak > memory_max_bytes:
        raise CaptureError("WSL2 scope memory peak exceeded its maximum")
    if pids_peak > benchmark.WSL2_SCOPE_PIDS_MAX:
        raise CaptureError("WSL2 scope PID peak exceeded its maximum")
    if Path(expected_cgroup).exists():
        raise CaptureError("WSL2 scope cgroup still exists after completion")
    memory_events = complete.get("memory_events")
    if not isinstance(memory_events, dict) or not memory_events:
        raise CaptureError("WSL2 scope memory events are invalid")
    if any(
        not isinstance(name, str)
        or not name
        or isinstance(value, bool)
        or not isinstance(value, int)
        or value < 0
        for name, value in memory_events.items()
    ):
        raise CaptureError("WSL2 scope memory event counters are invalid")
    for field in ("max", "oom", "oom_kill", "oom_group_kill"):
        if memory_events.get(field) != 0:
            raise CaptureError(f"WSL2 scope memory event {field} was not zero")
    return {
        "mechanism": (
            "systemd-user-scope-v1"
            if thermal is None
            else "systemd-user-scope-feedback-v1"
        ),
        "unit": unit,
        "cgroup": expected_cgroup,
        "network_containment": benchmark.WSL2_NETWORK_BOUNDARY,
        "memory_max_bytes": memory_max_bytes,
        "memory_swap_max_bytes": 0,
        "memory_peak_bytes": memory_peak,
        "memory_events": memory_events,
        "pids_max": benchmark.WSL2_SCOPE_PIDS_MAX,
        "pids_peak": pids_peak,
        "cpu_quota_percent": cpu_quota_percent,
        "cpu_poll_interval_ms": WSL_SCOPE_CPU_POLL_INTERVAL_MS,
        "cpu_usage_usec": cpu_usage,
        "cpu_allowed_usec": cpu_allowed,
        "duration_seconds": duration_seconds,
        "scope_removed": True,
        "thermal_pacing": pacing_evidence,
    }


def validate_wsl2_event_order(
    stderr_payload: bytes,
    *,
    thermal_requested: bool,
) -> None:
    observed: list[tuple[str, Any]] = []
    for raw_line in stderr_payload.decode("utf-8", errors="replace").splitlines():
        for source, prefix in (
            ("thermal", WSL_THERMAL_EVENT_PREFIX),
            ("scope", WSL_SCOPE_EVENT_PREFIX),
        ):
            if not raw_line.startswith(prefix):
                continue
            payload = raw_line.removeprefix(prefix).encode("utf-8")
            try:
                event = benchmark.strict_json_loads(payload)
            except Exception as exc:
                raise CaptureError(
                    f"WSL2 {source} event order payload is invalid: {exc}"
                ) from exc
            observed.append(
                (
                    source,
                    event.get("event") if isinstance(event, dict) else None,
                )
            )
            break
    expected = (
        [
            ("thermal", "preflight"),
            ("scope", "start"),
            ("scope", "complete"),
            ("thermal", "complete"),
        ]
        if thermal_requested
        else [
            ("scope", "start"),
            ("scope", "complete"),
        ]
    )
    if observed != expected:
        raise CaptureError("WSL2 thermal/scope event order is invalid")


def capture_once(
    config: Any,
    *,
    timeout_seconds: float,
    wsl2_supervision: Wsl2ScopeSupervision | None = None,
) -> Capture:
    """Execute one bounded manifest-only child and validate its exact output."""

    command = supervised_manifest_command(config, wsl2_supervision, timeout_seconds)
    termination_grace_seconds = CAPTURE_TERMINATION_GRACE_SECONDS
    environment: dict[str, str] | None = None
    scope_runtime_seconds: float | None = None
    thermal = (
        None if wsl2_supervision is None else wsl2_supervision.thermal
    )
    if thermal is not None:
        termination_grace_seconds += thermal.policy.handoff_timeout_seconds
    if wsl2_supervision is not None:
        scope_runtime_seconds = (
            timeout_seconds
            - (
                0.0
                if thermal is None
                else thermal.policy.handoff_timeout_seconds
            )
            - CAPTURE_TERMINATION_GRACE_SECONDS
        )
        environment = dict(os.environ)
        environment[benchmark.wsl_platform.NETWORK_ISOLATION_ENV] = (
            benchmark.WSL2_NETWORK_BOUNDARY
        )
    with tempfile.TemporaryDirectory(prefix="kiln-vllm-manifest-") as directory:
        capture_root = Path(directory)
        stdout_path = capture_root / "stdout"
        stderr_path = capture_root / "stderr"
        with stdout_path.open("xb") as stdout, stderr_path.open("xb") as stderr:
            returncode = _run_capture_child(
                command,
                working_directory=config.working_directory,
                stdout=stdout,
                stderr=stderr,
                timeout_seconds=timeout_seconds,
                termination_grace_seconds=termination_grace_seconds,
                environment=environment,
            )
        stderr_bytes = stderr_path.stat().st_size
        stderr_payload = stderr_path.read_bytes()
        if returncode != 0:
            details = stderr_payload.decode("utf-8", errors="replace")[-4096:].strip()
            raise CaptureError(
                f"vLLM runtime manifest child exited {returncode}: {details}"
            )
        payload = stdout_path.read_bytes()
    if not payload or len(payload) > MAX_MANIFEST_BYTES:
        raise CaptureError(
            f"vLLM runtime manifest output must be in 1..={MAX_MANIFEST_BYTES} bytes"
        )
    try:
        value = benchmark.strict_json_loads(payload)
        manifest = benchmark.validate_vllm_runtime_manifest(
            value,
            "captured vLLM runtime manifest",
        )
        benchmark.validate_vllm_owned_launch(config, manifest)
    except Exception as exc:
        raise CaptureError(f"captured vLLM runtime manifest is invalid: {exc}") from exc
    thermal_evidence = (
        None
        if thermal is None
        else validate_wsl2_thermal_stderr(stderr_payload, thermal)
    )
    if wsl2_supervision is not None and scope_runtime_seconds is None:
        raise CaptureError("WSL2 scope runtime bound was not derived")
    scope_evidence = (
        None
        if wsl2_supervision is None
        else validate_wsl2_scope_stderr(
            stderr_payload,
            wsl2_supervision,
            scope_runtime_seconds,
        )
    )
    if wsl2_supervision is not None:
        validate_wsl2_event_order(
            stderr_payload,
            thermal_requested=thermal is not None,
        )
    return Capture(
        payload=payload,
        manifest=manifest,
        stderr_bytes=stderr_bytes,
        stderr_sha256="sha256:" + hashlib.sha256(stderr_payload).hexdigest(),
        wsl2_thermal=thermal_evidence,
        wsl2_scope=scope_evidence,
    )


def capture_twice(
    config: Any,
    *,
    timeout_seconds: float,
    wsl2_supervision: Wsl2ScopeSupervision | None = None,
) -> tuple[Capture, Capture]:
    """Require two byte-identical runtime and accelerator observations."""

    first = capture_once(
        config,
        timeout_seconds=timeout_seconds,
        wsl2_supervision=wsl2_supervision,
    )
    second = capture_once(
        config,
        timeout_seconds=timeout_seconds,
        wsl2_supervision=wsl2_supervision,
    )
    if first.payload != second.payload:
        raise CaptureError(
            "two vLLM runtime manifest captures were not byte-identical: "
            f"sha256:{hashlib.sha256(first.payload).hexdigest()} != "
            f"sha256:{hashlib.sha256(second.payload).hexdigest()}"
        )
    return first, second


def publish_no_clobber(path: Path, payload: bytes) -> None:
    """Durably publish the exact repeated bytes without replacing any path."""

    if path.exists() or path.is_symlink():
        raise CaptureError(f"refusing to replace existing runtime manifest: {path}")
    if not path.parent.is_dir():
        raise CaptureError(
            f"runtime manifest output parent is not a directory: {path.parent}"
        )
    temporary_path: Path | None = None
    try:
        descriptor, temporary = tempfile.mkstemp(
            prefix=f".{path.name}.",
            suffix=".tmp",
            dir=path.parent,
        )
        temporary_path = Path(temporary)
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            os.fchmod(handle.fileno(), 0o644)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary_path, path, follow_symlinks=False)
        except FileExistsError as exc:
            raise CaptureError(
                f"refusing to replace existing runtime manifest: {path}"
            ) from exc
        directory_descriptor = os.open(
            path.parent,
            os.O_RDONLY | getattr(os, "O_CLOEXEC", 0),
        )
        try:
            os.fsync(directory_descriptor)
        finally:
            os.close(directory_descriptor)
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--server-launch-config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--wsl2-thermal-policy",
        type=Path,
        help=(
            "optional content-hashed repository v2 lab policy used to pace, "
            "supervise, and cool each capture independently"
        ),
    )
    parser.add_argument(
        "--timeout-seconds",
        type=float,
        default=DEFAULT_TIMEOUT_SECONDS,
        help=f"deadline for each of two captures (default: {DEFAULT_TIMEOUT_SECONDS:g})",
    )
    args = parser.parse_args(argv)
    if not 1.0 <= args.timeout_seconds <= 7200.0:
        parser.error("timeout-seconds must be in 1..=7200")
    return args


def main(argv: Sequence[str] | None = None) -> int:
    try:
        args = parse_args(argv)
        if args.output.exists() or args.output.is_symlink():
            raise CaptureError(
                f"refusing to replace existing runtime manifest: {args.output}"
            )
        commit = require_clean_repository()
        launch_path, _launch_repository_path = _repository_regular_file(
            args.server_launch_config,
            "server launch config",
        )
        config = benchmark.load_server_launch_config(launch_path)
        wsl2_supervision = load_platform_supervision(args.wsl2_thermal_policy)
        first, second = capture_twice(
            config,
            timeout_seconds=args.timeout_seconds,
            wsl2_supervision=wsl2_supervision,
        )
        if require_clean_repository() != commit:
            raise CaptureError(
                "repository commit changed during vLLM manifest capture"
            )
        publish_no_clobber(args.output, first.payload)
        result = {
            "capture_count": 2,
            "manifest_bytes": len(first.payload),
            "manifest_sha256": "sha256:" + hashlib.sha256(first.payload).hexdigest(),
            "output": str(args.output.absolute()),
            "runtime_content_sha256": first.manifest["runtime_content_sha256"],
            "source_commit": commit,
            "stderr": [
                {
                    "bytes": first.stderr_bytes,
                    "sha256": first.stderr_sha256,
                },
                {
                    "bytes": second.stderr_bytes,
                    "sha256": second.stderr_sha256,
                },
            ],
            "system_fingerprint": first.manifest["system_fingerprint"],
            "wsl2_thermal_supervision": (
                None
                if (
                    wsl2_supervision is None
                    or wsl2_supervision.thermal is None
                )
                else [first.wsl2_thermal, second.wsl2_thermal]
            ),
            "wsl2_scope_supervision": (
                None
                if wsl2_supervision is None
                else [first.wsl2_scope, second.wsl2_scope]
            ),
        }
        print(
            json.dumps(
                result,
                allow_nan=False,
                ensure_ascii=True,
                separators=(",", ":"),
                sort_keys=True,
            )
        )
        return 0
    except (CaptureError, benchmark.BenchmarkError, OSError, subprocess.SubprocessError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
