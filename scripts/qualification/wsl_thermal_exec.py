#!/usr/bin/env python3
"""Run one WSL2 qualification case under host and GPU thermal supervision."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import re
import signal
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence


SCHEMA_V1 = "kiln.wsl2-thermal-policy.v1"
SCHEMA_V2 = "kiln.wsl2-thermal-policy.v2"
SCHEMA = SCHEMA_V1
SCHEMAS = frozenset({SCHEMA_V1, SCHEMA_V2})
POLICY_ENV = "KILN_WSL2_THERMAL_POLICY_SHA256"
PACING_POLICY_ENV = "KILN_WSL2_CGROUP_THERMAL_PACING_POLICY_SHA256"
TELEMETRY_FD_ENV = "KILN_WSL2_THERMAL_TELEMETRY_FD"
TELEMETRY_SCHEMA = "kiln.wsl2-thermal-sample.v1"
POWERSHELL = Path("/mnt/c/Windows/System32/WindowsPowerShell/v1.0/powershell.exe")
NVIDIA_SMI = Path("/usr/lib/wsl/lib/nvidia-smi")
SHA256_RE = re.compile(r"^sha256:[0-9a-f]{64}$")
POLICY_KEYS = {
    "schema",
    "id",
    "content_sha256",
    "host",
    "gpu",
    "poll_interval_ms",
    "safe_handoff",
}
POLICY_V2_KEYS = POLICY_KEYS | {"pacing"}
HOST_KEYS = {
    "cpu_name",
    "thermal_zone_name",
    "limit_millicelsius",
    "vendor_tjunction_millicelsius",
}
GPU_KEYS = {"name", "uuid", "limit_millicelsius"}
HANDOFF_KEYS = {
    "host_target_millicelsius",
    "gpu_target_millicelsius",
    "stable_samples",
    "timeout_seconds",
}
PACING_KEYS = {
    "mode",
    "host_start_millicelsius",
    "host_resume_millicelsius",
    "gpu_start_millicelsius",
    "gpu_resume_millicelsius",
    "resume_stable_samples",
    "timeout_seconds",
}


class ThermalGuardError(RuntimeError):
    """The WSL2 thermal policy or live telemetry failed closed."""


@dataclass(frozen=True)
class ThermalPacingPolicy:
    mode: str
    host_start_millicelsius: int
    host_resume_millicelsius: int
    gpu_start_millicelsius: int
    gpu_resume_millicelsius: int
    resume_stable_samples: int
    timeout_seconds: float


@dataclass(frozen=True)
class ThermalPolicy:
    schema: str
    policy_id: str
    content_sha256: str
    cpu_name: str
    thermal_zone_name: str
    host_limit_millicelsius: int
    vendor_tjunction_millicelsius: int
    gpu_name: str
    gpu_uuid: str
    gpu_limit_millicelsius: int
    poll_interval_ms: int
    handoff_host_millicelsius: int
    handoff_gpu_millicelsius: int
    handoff_stable_samples: int
    handoff_timeout_seconds: float
    pacing: ThermalPacingPolicy | None


@dataclass(frozen=True)
class ThermalSample:
    monotonic_seconds: float
    host_millicelsius: int
    gpu_millicelsius: int


@dataclass(frozen=True)
class HandoffResult:
    ending: ThermalSample
    sample_count: int
    peak_host_millicelsius: int
    peak_gpu_millicelsius: int
    limit_failure: ThermalGuardError | None


def _exact_object(value: Any, keys: set[str], context: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ThermalGuardError(f"{context} must be an object")
    missing = sorted(keys - value.keys())
    unknown = sorted(value.keys() - keys)
    if missing or unknown:
        details: list[str] = []
        if missing:
            details.append("missing " + ", ".join(missing))
        if unknown:
            details.append("unknown " + ", ".join(unknown))
        raise ThermalGuardError(f"{context} has invalid keys: {'; '.join(details)}")
    return value


def _string(value: Any, context: str) -> str:
    if not isinstance(value, str) or not value or any(
        character in value for character in ("\0", "\r", "\n")
    ):
        raise ThermalGuardError(f"{context} must be a nonempty single-line string")
    return value


def _integer(value: Any, context: str, minimum: int, maximum: int) -> int:
    if (
        not isinstance(value, int)
        or isinstance(value, bool)
        or value < minimum
        or value > maximum
    ):
        raise ThermalGuardError(f"{context} must be in {minimum}..={maximum}")
    return value


def _number(value: Any, context: str, minimum: float, maximum: float) -> float:
    if (
        not isinstance(value, (int, float))
        or isinstance(value, bool)
        or not math.isfinite(value)
        or float(value) < minimum
        or float(value) > maximum
    ):
        raise ThermalGuardError(f"{context} must be in {minimum:g}..={maximum:g}")
    return float(value)


def _canonical_policy_hash(raw: dict[str, Any]) -> str:
    content = dict(raw)
    content.pop("content_sha256", None)
    payload = json.dumps(
        content,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("ascii")
    return f"sha256:{hashlib.sha256(payload).hexdigest()}"


def validate_policy(raw: Any) -> ThermalPolicy:
    if not isinstance(raw, dict):
        raise ThermalGuardError("WSL2 thermal policy must be an object")
    schema = raw.get("schema")
    if schema not in SCHEMAS:
        raise ThermalGuardError(
            "WSL2 thermal policy schema must be one of "
            + ", ".join(repr(value) for value in sorted(SCHEMAS))
        )
    root = _exact_object(
        raw,
        POLICY_KEYS if schema == SCHEMA_V1 else POLICY_V2_KEYS,
        "WSL2 thermal policy",
    )
    policy_id = _string(root["id"], "policy id")
    content_sha256 = _string(root["content_sha256"], "policy content_sha256")
    if not SHA256_RE.fullmatch(content_sha256):
        raise ThermalGuardError("policy content_sha256 must be lowercase sha256")
    calculated = _canonical_policy_hash(root)
    if content_sha256 != calculated:
        raise ThermalGuardError(
            f"policy content hash mismatch: declared {content_sha256}, calculated {calculated}"
        )

    host = _exact_object(root["host"], HOST_KEYS, "policy host")
    gpu = _exact_object(root["gpu"], GPU_KEYS, "policy gpu")
    handoff = _exact_object(root["safe_handoff"], HANDOFF_KEYS, "policy safe_handoff")
    host_limit = _integer(
        host["limit_millicelsius"],
        "host limit_millicelsius",
        1,
        200_000,
    )
    tjunction = _integer(
        host["vendor_tjunction_millicelsius"],
        "host vendor_tjunction_millicelsius",
        1,
        200_000,
    )
    if host_limit >= tjunction:
        raise ThermalGuardError("host limit must be below the vendor Tjunction")
    gpu_limit = _integer(
        gpu["limit_millicelsius"],
        "gpu limit_millicelsius",
        1,
        200_000,
    )
    host_target = _integer(
        handoff["host_target_millicelsius"],
        "safe_handoff host target",
        1,
        200_000,
    )
    gpu_target = _integer(
        handoff["gpu_target_millicelsius"],
        "safe_handoff gpu target",
        1,
        200_000,
    )
    if host_target >= host_limit or gpu_target >= gpu_limit:
        raise ThermalGuardError("safe-handoff targets must be below their hard limits")
    pacing: ThermalPacingPolicy | None = None
    if schema == SCHEMA_V2:
        pacing_value = _exact_object(root["pacing"], PACING_KEYS, "policy pacing")
        if pacing_value["mode"] != "cgroup_freeze":
            raise ThermalGuardError("policy pacing mode must be 'cgroup_freeze'")
        host_start = _integer(
            pacing_value["host_start_millicelsius"],
            "pacing host start",
            1,
            200_000,
        )
        host_resume = _integer(
            pacing_value["host_resume_millicelsius"],
            "pacing host resume",
            1,
            200_000,
        )
        gpu_start = _integer(
            pacing_value["gpu_start_millicelsius"],
            "pacing GPU start",
            1,
            200_000,
        )
        gpu_resume = _integer(
            pacing_value["gpu_resume_millicelsius"],
            "pacing GPU resume",
            1,
            200_000,
        )
        if not host_resume < host_start < host_limit:
            raise ThermalGuardError(
                "pacing host resume must be below start, and start below the hard limit"
            )
        if not gpu_resume < gpu_start < gpu_limit:
            raise ThermalGuardError(
                "pacing GPU resume must be below start, and start below the hard limit"
            )
        if host_resume > host_target or gpu_resume > gpu_target:
            raise ThermalGuardError(
                "pacing resume targets must not exceed safe-handoff targets"
            )
        pacing = ThermalPacingPolicy(
            mode="cgroup_freeze",
            host_start_millicelsius=host_start,
            host_resume_millicelsius=host_resume,
            gpu_start_millicelsius=gpu_start,
            gpu_resume_millicelsius=gpu_resume,
            resume_stable_samples=_integer(
                pacing_value["resume_stable_samples"],
                "pacing resume_stable_samples",
                1,
                10_000,
            ),
            timeout_seconds=_number(
                pacing_value["timeout_seconds"],
                "pacing timeout_seconds",
                1.0,
                3600.0,
            ),
        )
    return ThermalPolicy(
        schema=schema,
        policy_id=policy_id,
        content_sha256=content_sha256,
        cpu_name=_string(host["cpu_name"], "host cpu_name"),
        thermal_zone_name=_string(host["thermal_zone_name"], "host thermal_zone_name"),
        host_limit_millicelsius=host_limit,
        vendor_tjunction_millicelsius=tjunction,
        gpu_name=_string(gpu["name"], "gpu name"),
        gpu_uuid=_string(gpu["uuid"], "gpu uuid"),
        gpu_limit_millicelsius=gpu_limit,
        poll_interval_ms=_integer(
            root["poll_interval_ms"],
            "poll_interval_ms",
            100,
            60_000,
        ),
        handoff_host_millicelsius=host_target,
        handoff_gpu_millicelsius=gpu_target,
        handoff_stable_samples=_integer(
            handoff["stable_samples"],
            "safe_handoff stable_samples",
            1,
            10_000,
        ),
        handoff_timeout_seconds=_number(
            handoff["timeout_seconds"],
            "safe_handoff timeout_seconds",
            1.0,
            3600.0,
        ),
        pacing=pacing,
    )


def load_policy(path: Path) -> ThermalPolicy:
    try:
        raw = json.loads(path.read_bytes())
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ThermalGuardError(f"cannot read WSL2 thermal policy {path}: {exc}") from exc
    return validate_policy(raw)


def _run(command: list[str], label: str, timeout: float = 10.0) -> str:
    try:
        completed = subprocess.run(
            command,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
            timeout=timeout,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise ThermalGuardError(f"{label} failed: {exc}") from exc
    stdout = completed.stdout.decode("utf-8-sig", errors="strict").strip()
    stderr = completed.stderr.decode("utf-8-sig", errors="replace").strip()
    if completed.returncode != 0:
        raise ThermalGuardError(
            f"{label} exited {completed.returncode}"
            + (f": {stderr}" if stderr else "")
        )
    if not stdout:
        raise ThermalGuardError(f"{label} returned empty output")
    return stdout


def _powershell_json(script: str, label: str) -> Any:
    text = _run(
        [
            str(POWERSHELL),
            "-NoLogo",
            "-NoProfile",
            "-NonInteractive",
            "-Command",
            "$ErrorActionPreference='Stop'; " + script,
        ],
        label,
    )
    try:
        return json.loads(text)
    except json.JSONDecodeError as exc:
        raise ThermalGuardError(f"{label} returned malformed JSON: {exc}") from exc


def verify_host_identity(policy: ThermalPolicy) -> None:
    if "microsoft-standard-wsl2" not in platform.release().lower():
        raise ThermalGuardError("WSL2 thermal supervision requires a WSL2 kernel")
    if not POWERSHELL.is_file():
        raise ThermalGuardError(f"Windows PowerShell is unavailable: {POWERSHELL}")
    if not NVIDIA_SMI.is_file():
        raise ThermalGuardError(f"WSL NVIDIA SMI is unavailable: {NVIDIA_SMI}")
    value = _powershell_json(
        "(Get-CimInstance -ClassName Win32_Processor | "
        "Select-Object -ExpandProperty Name) | ConvertTo-Json -Compress",
        "Windows CPU identity probe",
    )
    if value != policy.cpu_name:
        raise ThermalGuardError(
            f"CPU identity mismatch: observed {value!r}, expected {policy.cpu_name!r}"
        )


def _host_temperature(policy: ThermalPolicy) -> int:
    raw = _powershell_json(
        "Get-CimInstance -ClassName "
        "Win32_PerfFormattedData_Counters_ThermalZoneInformation | "
        "Select-Object Name,Temperature,HighPrecisionTemperature,"
        "PercentPassiveLimit,ThrottleReasons | ConvertTo-Json -Compress",
        "Windows thermal-zone probe",
    )
    rows = raw if isinstance(raw, list) else [raw]
    matches = [
        row
        for row in rows
        if isinstance(row, dict) and row.get("Name") == policy.thermal_zone_name
    ]
    if len(matches) != 1:
        raise ThermalGuardError(
            f"thermal zone {policy.thermal_zone_name!r} matched {len(matches)} rows"
        )
    row = _exact_object(
        matches[0],
        {
            "Name",
            "Temperature",
            "HighPrecisionTemperature",
            "PercentPassiveLimit",
            "ThrottleReasons",
        },
        "Windows thermal-zone row",
    )
    kelvin = _integer(row["Temperature"], "thermal Temperature", 1, 1000)
    tenths_kelvin = _integer(
        row["HighPrecisionTemperature"],
        "thermal HighPrecisionTemperature",
        1,
        10_000,
    )
    if abs(kelvin * 10 - tenths_kelvin) > 10:
        raise ThermalGuardError(
            "Windows thermal Temperature and HighPrecisionTemperature disagree"
        )
    temperature = tenths_kelvin * 100 - 273_150
    if temperature < -50_000 or temperature > 200_000:
        raise ThermalGuardError(
            f"Windows thermal zone returned implausible {temperature} millicelsius"
        )
    return temperature


def _gpu_temperature(policy: ThermalPolicy) -> int:
    text = _run(
        [
            str(NVIDIA_SMI),
            "--query-gpu=name,uuid,temperature.gpu",
            "--format=csv,noheader,nounits",
        ],
        "WSL NVML temperature probe",
    )
    rows = [row.strip() for row in text.splitlines() if row.strip()]
    parsed: list[tuple[str, str, int]] = []
    for row in rows:
        fields = [field.strip() for field in row.split(",")]
        if len(fields) != 3 or not re.fullmatch(r"[0-9]+", fields[2]):
            raise ThermalGuardError(f"malformed WSL NVML temperature row: {row!r}")
        parsed.append((fields[0], fields[1], int(fields[2])))
    matches = [row for row in parsed if row[1] == policy.gpu_uuid]
    if len(matches) != 1:
        raise ThermalGuardError(
            f"GPU UUID {policy.gpu_uuid!r} matched {len(matches)} devices"
        )
    name, _uuid, celsius = matches[0]
    if name != policy.gpu_name:
        raise ThermalGuardError(
            f"GPU identity mismatch: observed {name!r}, expected {policy.gpu_name!r}"
        )
    if celsius <= 0 or celsius > 200:
        raise ThermalGuardError(f"GPU returned implausible {celsius} C")
    return celsius * 1000


def sample(policy: ThermalPolicy) -> ThermalSample:
    return ThermalSample(
        monotonic_seconds=time.monotonic(),
        host_millicelsius=_host_temperature(policy),
        gpu_millicelsius=_gpu_temperature(policy),
    )


def _emit(event: str, policy: ThermalPolicy, **fields: Any) -> None:
    value = {
        "schema": "kiln.wsl2-thermal-event.v1",
        "event": event,
        "policy_id": policy.policy_id,
        "policy_sha256": policy.content_sha256,
        **fields,
    }
    print(
        "wsl2-thermal: "
        + json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        ),
        file=sys.stderr,
        flush=True,
    )


def _send_telemetry_sample(
    descriptor: int,
    policy: ThermalPolicy,
    sequence: int,
    value: ThermalSample,
) -> None:
    payload = (
        json.dumps(
            {
                "schema": TELEMETRY_SCHEMA,
                "policy_sha256": policy.content_sha256,
                "sequence": sequence,
                "monotonic_seconds": value.monotonic_seconds,
                "host_millicelsius": value.host_millicelsius,
                "gpu_millicelsius": value.gpu_millicelsius,
            },
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        )
        + "\n"
    ).encode("ascii")
    try:
        written = os.write(descriptor, payload)
    except (BlockingIOError, BrokenPipeError, OSError) as exc:
        raise ThermalGuardError(
            f"cannot deliver thermal telemetry to the scope controller: {exc}"
        ) from exc
    if written != len(payload):
        raise ThermalGuardError(
            "thermal telemetry pipe accepted only a partial sample"
        )


def _check_limits(value: ThermalSample, policy: ThermalPolicy) -> None:
    if value.host_millicelsius >= policy.host_limit_millicelsius:
        raise ThermalGuardError(
            f"host temperature {value.host_millicelsius} reached "
            f"{policy.host_limit_millicelsius} millicelsius"
        )
    if value.gpu_millicelsius >= policy.gpu_limit_millicelsius:
        raise ThermalGuardError(
            f"GPU temperature {value.gpu_millicelsius} reached "
            f"{policy.gpu_limit_millicelsius} millicelsius"
        )


def _terminate(process: subprocess.Popen[Any], grace_seconds: float = 15.0) -> None:
    if process.poll() is not None:
        return
    process.terminate()
    deadline = time.monotonic() + grace_seconds
    while process.poll() is None and time.monotonic() < deadline:
        time.sleep(0.05)
    if process.poll() is None:
        process.kill()
    process.wait()


def _safe_handoff(
    policy: ThermalPolicy,
    *,
    first: ThermalSample | None = None,
) -> HandoffResult:
    deadline = time.monotonic() + policy.handoff_timeout_seconds
    stable = 0
    current = first
    sample_count = 0
    peak_host = 0
    peak_gpu = 0
    limit_failure: ThermalGuardError | None = None
    while True:
        current = current or sample(policy)
        sample_count += 1
        peak_host = max(peak_host, current.host_millicelsius)
        peak_gpu = max(peak_gpu, current.gpu_millicelsius)
        try:
            _check_limits(current, policy)
        except ThermalGuardError as exc:
            if limit_failure is None:
                limit_failure = exc
        below = (
            current.host_millicelsius <= policy.handoff_host_millicelsius
            and current.gpu_millicelsius <= policy.handoff_gpu_millicelsius
        )
        stable = stable + 1 if below else 0
        if stable >= policy.handoff_stable_samples:
            return HandoffResult(
                ending=current,
                sample_count=sample_count,
                peak_host_millicelsius=peak_host,
                peak_gpu_millicelsius=peak_gpu,
                limit_failure=limit_failure,
            )
        if time.monotonic() >= deadline:
            raise ThermalGuardError(
                "safe handoff timed out: "
                f"host={current.host_millicelsius}, gpu={current.gpu_millicelsius}"
            )
        time.sleep(policy.poll_interval_ms / 1000.0)
        current = None


def _validate_pacing_scope_command(
    policy: ThermalPolicy,
    command: Sequence[str],
) -> None:
    if policy.pacing is None:
        return
    trusted_scope = Path(__file__).with_name("wsl_scope_exec.py")
    if (
        len(command) < 5
        or command[0] != sys.executable
        or Path(command[1]).is_symlink()
    ):
        raise ThermalGuardError(
            "a pacing policy requires the trusted WSL2 scope supervisor"
        )
    try:
        scope_matches = Path(command[1]).samefile(trusted_scope)
    except OSError as exc:
        raise ThermalGuardError(
            f"cannot verify the WSL2 scope supervisor: {exc}"
        ) from exc
    if not scope_matches:
        raise ThermalGuardError(
            "a pacing policy requires the trusted WSL2 scope supervisor"
        )
    try:
        scope_boundary = command.index("--", 2)
    except ValueError as exc:
        raise ThermalGuardError(
            "the WSL2 scope supervisor command has no argument boundary"
        ) from exc
    positions = [
        index
        for index, value in enumerate(command[2:scope_boundary], start=2)
        if value == "--thermal-pacing-policy"
    ]
    if len(positions) != 1 or positions[0] + 1 >= scope_boundary:
        raise ThermalGuardError(
            "the WSL2 scope supervisor must receive exactly one thermal pacing policy"
        )
    pacing_path = Path(command[positions[0] + 1])
    if pacing_path.is_symlink() or not pacing_path.is_file():
        raise ThermalGuardError(
            "the WSL2 scope thermal pacing policy must be a regular non-symlink file"
        )
    pacing_policy = load_policy(pacing_path)
    if (
        pacing_policy.pacing is None
        or pacing_policy.content_sha256 != policy.content_sha256
    ):
        raise ThermalGuardError(
            "the WSL2 scope thermal pacing policy does not match the outer policy"
        )


def supervise(policy: ThermalPolicy, command: Sequence[str]) -> int:
    if not command:
        raise ThermalGuardError("a supervised command is required")
    _validate_pacing_scope_command(policy, command)
    verify_host_identity(policy)
    starting = sample(policy)
    _check_limits(starting, policy)
    _emit(
        "preflight",
        policy,
        host_millicelsius=starting.host_millicelsius,
        gpu_millicelsius=starting.gpu_millicelsius,
        host_limit_millicelsius=policy.host_limit_millicelsius,
        gpu_limit_millicelsius=policy.gpu_limit_millicelsius,
    )
    child_environment = dict(os.environ)
    child_environment[POLICY_ENV] = policy.content_sha256
    telemetry_read_fd: int | None = None
    telemetry_write_fd: int | None = None
    if policy.pacing is not None:
        child_environment[PACING_POLICY_ENV] = policy.content_sha256
        telemetry_read_fd, telemetry_write_fd = os.pipe2(os.O_CLOEXEC)
        os.set_blocking(telemetry_write_fd, False)
        child_environment[TELEMETRY_FD_ENV] = str(telemetry_read_fd)
    try:
        process = subprocess.Popen(
            list(command),
            env=child_environment,
            pass_fds=(
                ()
                if telemetry_read_fd is None
                else (telemetry_read_fd,)
            ),
        )
    except OSError as exc:
        if telemetry_read_fd is not None:
            os.close(telemetry_read_fd)
        if telemetry_write_fd is not None:
            os.close(telemetry_write_fd)
        raise ThermalGuardError(f"cannot launch supervised command: {exc}") from exc
    if telemetry_read_fd is not None:
        os.close(telemetry_read_fd)
    telemetry_sequence = 0
    if telemetry_write_fd is not None:
        try:
            _send_telemetry_sample(
                telemetry_write_fd,
                policy,
                telemetry_sequence,
                starting,
            )
        except ThermalGuardError:
            _terminate(process)
            os.close(telemetry_write_fd)
            raise

    interrupted_signal: int | None = None

    def interrupt(signum: int, _frame: Any) -> None:
        nonlocal interrupted_signal
        interrupted_signal = signum

    old_handlers = {
        signum: signal.signal(signum, interrupt)
        for signum in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP)
    }
    peak_host = starting.host_millicelsius
    peak_gpu = starting.gpu_millicelsius
    samples = 1
    failure: ThermalGuardError | None = None
    handoff: HandoffResult
    try:
        while process.poll() is None:
            if interrupted_signal is not None:
                failure = ThermalGuardError(
                    f"supervisor received signal {interrupted_signal}"
                )
                break
            time.sleep(policy.poll_interval_ms / 1000.0)
            try:
                current = sample(policy)
                samples += 1
                peak_host = max(peak_host, current.host_millicelsius)
                peak_gpu = max(peak_gpu, current.gpu_millicelsius)
                _check_limits(current, policy)
            except ThermalGuardError as exc:
                failure = exc
                break
            telemetry_sequence += 1
            if telemetry_write_fd is not None:
                try:
                    _send_telemetry_sample(
                        telemetry_write_fd,
                        policy,
                        telemetry_sequence,
                        current,
                    )
                except ThermalGuardError as exc:
                    if process.poll() is None:
                        failure = exc
                    break
        if failure is not None:
            _emit("trip", policy, reason=str(failure))
            _terminate(process)
        returncode = process.wait()
        try:
            handoff = _safe_handoff(policy)
        except ThermalGuardError as exc:
            if failure is not None:
                raise ThermalGuardError(
                    f"{failure}; safe handoff failed: {exc}"
                ) from exc
            raise
    finally:
        for signum, handler in old_handlers.items():
            signal.signal(signum, handler)
        if telemetry_write_fd is not None:
            os.close(telemetry_write_fd)

    samples += handoff.sample_count
    peak_host = max(peak_host, handoff.peak_host_millicelsius)
    peak_gpu = max(peak_gpu, handoff.peak_gpu_millicelsius)
    if handoff.limit_failure is not None and failure is None:
        failure = handoff.limit_failure
        _emit("trip", policy, reason=str(failure))
    outcome = (
        "interrupted"
        if interrupted_signal is not None
        else "thermal_trip"
        if failure is not None
        else "child_exit"
    )
    _emit(
        "complete",
        policy,
        supervision_outcome=outcome,
        failure_reason=None if failure is None else str(failure),
        child_returncode=returncode,
        sample_count=samples,
        starting_host_millicelsius=starting.host_millicelsius,
        starting_gpu_millicelsius=starting.gpu_millicelsius,
        peak_host_millicelsius=peak_host,
        peak_gpu_millicelsius=peak_gpu,
        ending_host_millicelsius=handoff.ending.host_millicelsius,
        ending_gpu_millicelsius=handoff.ending.gpu_millicelsius,
        safe_handoff_stable_samples=policy.handoff_stable_samples,
    )
    if interrupted_signal is not None:
        return 128 + interrupted_signal
    if failure is not None:
        raise failure
    return returncode


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy", type=Path, required=True)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args(argv)
    if args.command and args.command[0] == "--":
        args.command = args.command[1:]
    if not args.command:
        parser.error("a command is required after --")
    return args


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        policy = load_policy(args.policy)
        return supervise(policy, args.command)
    except ThermalGuardError as exc:
        print(f"error: WSL2 thermal guard: {exc}", file=sys.stderr)
        return 3


if __name__ == "__main__":
    raise SystemExit(main())
