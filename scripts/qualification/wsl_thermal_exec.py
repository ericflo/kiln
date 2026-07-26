#!/usr/bin/env python3
"""Run one WSL2 qualification case under host and GPU thermal supervision."""

from __future__ import annotations

import argparse
import base64
import errno
import hashlib
import json
import math
import os
import platform
import re
import select
import signal
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Sequence

from device_memory_sampler import DeviceMemoryError, NvmlMemoryCounter


SCHEMA_V1 = "kiln.wsl2-thermal-policy.v1"
SCHEMA_V2 = "kiln.wsl2-thermal-policy.v2"
SCHEMA = SCHEMA_V1
SCHEMAS = frozenset({SCHEMA_V1, SCHEMA_V2})
POLICY_ENV = "KILN_WSL2_THERMAL_POLICY_SHA256"
PACING_POLICY_ENV = "KILN_WSL2_CGROUP_THERMAL_PACING_POLICY_SHA256"
TELEMETRY_FD_ENV = "KILN_WSL2_THERMAL_TELEMETRY_FD"
TELEMETRY_SCHEMA = "kiln.wsl2-thermal-sample.v1"
MAX_TELEMETRY_EXIT_SETTLEMENT_SECONDS = 1.0
POWERSHELL = Path("/mnt/c/Windows/System32/WindowsPowerShell/v1.0/powershell.exe")
NVIDIA_SMI = Path("/usr/lib/wsl/lib/nvidia-smi")
NVIDIA_NVML = Path("/usr/lib/wsl/lib/libnvidia-ml.so.1")
SHA256_RE = re.compile(r"^sha256:[0-9a-f]{64}$")
HOST_COUNTER_SCHEMA = "kiln.wsl2-host-thermal-counter.v2"
HOST_COUNTER_NAMES = (
    "Temperature",
    "High Precision Temperature",
    "% Passive Limit",
    "Throttle Reasons",
)
HOST_COUNTER_READY_KEYS = {"Schema", "CpuName", "Name", "CounterNames"}
HOST_COUNTER_SAMPLE_KEYS = {
    "Schema",
    "Sequence",
    "Name",
    "Timestamp100nSec",
    "Temperature",
    "HighPrecisionTemperature",
    "PercentPassiveLimit",
    "ThrottleReasons",
}
HOST_COUNTER_LINE_LIMIT = 4096
HOST_COUNTER_TIMEOUT_SECONDS = 10.0
HOST_COUNTER_SCRIPT = r"""
$ErrorActionPreference = 'Stop'
[Console]::InputEncoding = [System.Text.UTF8Encoding]::new($false)
[Console]::OutputEncoding = [System.Text.UTF8Encoding]::new($false)
$zone = [System.Text.Encoding]::UTF8.GetString(
    [System.Convert]::FromBase64String($zoneBase64)
)
$cpuKey = [Microsoft.Win32.Registry]::LocalMachine.OpenSubKey(
    'HARDWARE\DESCRIPTION\System\CentralProcessor\0'
)
if ($null -eq $cpuKey) {
    throw 'Windows CPU identity registry key is unavailable'
}
try {
    $cpuName = [string]$cpuKey.GetValue('ProcessorNameString')
} finally {
    $cpuKey.Dispose()
}
if ([string]::IsNullOrWhiteSpace($cpuName) -or $cpuName.Contains("`n") -or
    $cpuName.Contains("`r")) {
    throw 'Windows CPU identity registry value is invalid'
}
$category = [System.Diagnostics.PerformanceCounterCategory]::new(
    'Thermal Zone Information'
)
$instances = @($category.GetInstanceNames() | Where-Object { $_ -ceq $zone })
if ($instances.Count -ne 1) {
    throw "thermal zone '$zone' matched $($instances.Count) performance-counter instances"
}
$allCounters = @($category.GetCounters($zone))
function Resolve-Counter([string]$name) {
    $matches = @($allCounters | Where-Object { $_.CounterName -ceq $name })
    if ($matches.Count -ne 1) {
        throw "thermal counter '$name' matched $($matches.Count) counters"
    }
    return $matches[0]
}
foreach ($name in @(
    'Temperature',
    'High Precision Temperature',
    '% Passive Limit',
    'Throttle Reasons'
)) {
    Resolve-Counter $name | Out-Null
}
foreach ($counter in $allCounters) {
    $counter.Dispose()
}
function Read-SnapshotCounter($snapshot, [string]$name) {
    $counter = $snapshot[$name]
    if ($null -eq $counter) {
        throw "thermal snapshot omitted counter '$name'"
    }
    $instance = $counter[$zone]
    if ($null -eq $instance) {
        throw "thermal snapshot counter '$name' omitted exact instance '$zone'"
    }
    $sample = $instance.Sample
    if ([string]$sample.CounterType -cne 'NumberOfItems32') {
        throw "thermal snapshot counter '$name' has unexpected type '$($sample.CounterType)'"
    }
    return [ordered]@{
        Value = [long]$sample.RawValue
        Timestamp100nSec = [long]$sample.TimeStamp100nSec
    }
}
$writer = [Console]::Out
$reader = [Console]::In
$ready = [ordered]@{
    Schema = 'kiln.wsl2-host-thermal-counter.v2'
    CpuName = $cpuName
    Name = $zone
    CounterNames = @(
        'Temperature',
        'High Precision Temperature',
        '% Passive Limit',
        'Throttle Reasons'
    )
}
$writer.WriteLine(($ready | ConvertTo-Json -Compress))
$writer.Flush()
while ($null -ne ($line = $reader.ReadLine())) {
    if ($line -cnotmatch '^(0|[1-9][0-9]*)$') {
        throw "invalid thermal sample sequence '$line'"
    }
    $sequence = [long]::Parse(
        $line,
        [System.Globalization.CultureInfo]::InvariantCulture
    )
    $snapshot = $category.ReadCategory()
    $temperature = Read-SnapshotCounter $snapshot 'Temperature'
    $highPrecisionTemperature = Read-SnapshotCounter `
        $snapshot 'High Precision Temperature'
    $percentPassiveLimit = Read-SnapshotCounter $snapshot '% Passive Limit'
    $throttleReasons = Read-SnapshotCounter $snapshot 'Throttle Reasons'
    $timestamps = @(
        $temperature.Timestamp100nSec,
        $highPrecisionTemperature.Timestamp100nSec,
        $percentPassiveLimit.Timestamp100nSec,
        $throttleReasons.Timestamp100nSec
    ) | Select-Object -Unique
    if ($timestamps.Count -ne 1 -or $timestamps[0] -le 0) {
        throw 'thermal snapshot counter timestamps disagree or are invalid'
    }
    $sample = [ordered]@{
        Schema = 'kiln.wsl2-host-thermal-counter.v2'
        Sequence = $sequence
        Name = $zone
        Timestamp100nSec = [long]$timestamps[0]
        Temperature = $temperature.Value
        HighPrecisionTemperature = $highPrecisionTemperature.Value
        PercentPassiveLimit = $percentPassiveLimit.Value
        ThrottleReasons = $throttleReasons.Value
    }
    $writer.WriteLine(($sample | ConvertTo-Json -Compress))
    $writer.Flush()
}
"""
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


class ThermalTelemetryClosed(ThermalGuardError):
    """The owned scope closed its telemetry reader while exiting."""


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


@dataclass(frozen=True)
class PreflightResult:
    ending: ThermalSample
    sample_count: int
    peak_host_millicelsius: int
    peak_gpu_millicelsius: int


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


def verify_host_identity(_policy: ThermalPolicy) -> None:
    if "microsoft-standard-wsl2" not in platform.release().lower():
        raise ThermalGuardError("WSL2 thermal supervision requires a WSL2 kernel")
    if not POWERSHELL.is_file():
        raise ThermalGuardError(f"Windows PowerShell is unavailable: {POWERSHELL}")
    if not NVIDIA_SMI.is_file():
        raise ThermalGuardError(f"WSL NVIDIA SMI is unavailable: {NVIDIA_SMI}")
    if not NVIDIA_NVML.is_file():
        raise ThermalGuardError(f"WSL NVML is unavailable: {NVIDIA_NVML}")


def _parse_host_temperature(
    policy: ThermalPolicy,
    raw: Any,
) -> int:
    row = _exact_object(
        raw,
        {
            "Name",
            "Temperature",
            "HighPrecisionTemperature",
            "PercentPassiveLimit",
            "ThrottleReasons",
        },
        "Windows thermal-zone row",
    )
    if row["Name"] != policy.thermal_zone_name:
        raise ThermalGuardError(
            f"thermal zone identity mismatch: observed {row['Name']!r}, "
            f"expected {policy.thermal_zone_name!r}"
        )
    _integer(
        row["PercentPassiveLimit"],
        "thermal PercentPassiveLimit",
        0,
        100,
    )
    _integer(
        row["ThrottleReasons"],
        "thermal ThrottleReasons",
        0,
        2**32 - 1,
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
            "Windows thermal Temperature and HighPrecisionTemperature disagree: "
            f"Temperature={kelvin} K, "
            f"HighPrecisionTemperature={tenths_kelvin} tenths K"
        )
    temperature = tenths_kelvin * 100 - 273_150
    if temperature < -50_000 or temperature > 200_000:
        raise ThermalGuardError(
            f"Windows thermal zone returned implausible {temperature} millicelsius"
        )
    return temperature


def _strict_json_loads(text: str, label: str) -> Any:
    def closed_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        value: dict[str, Any] = {}
        for key, item in pairs:
            if key in value:
                raise ValueError(f"duplicate key {key!r}")
            value[key] = item
        return value

    try:
        return json.loads(text, object_pairs_hook=closed_object)
    except (json.JSONDecodeError, ValueError) as exc:
        raise ThermalGuardError(f"{label} returned malformed JSON: {exc}") from exc


class WindowsThermalCounter:
    """Persistent query-response channel to the Windows performance counters."""

    def __init__(
        self,
        policy: ThermalPolicy,
        *,
        process_factory: Callable[..., subprocess.Popen[bytes]] = subprocess.Popen,
    ) -> None:
        zone_base64 = base64.b64encode(
            policy.thermal_zone_name.encode("utf-8")
        ).decode("ascii")
        script = f"$zoneBase64 = '{zone_base64}'\n" + HOST_COUNTER_SCRIPT
        try:
            self._process = process_factory(
                [
                    str(POWERSHELL),
                    "-NoLogo",
                    "-NoProfile",
                    "-NonInteractive",
                    "-Command",
                    script,
                ],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                bufsize=0,
            )
        except OSError as exc:
            raise ThermalGuardError(
                f"cannot start persistent Windows thermal counter: {exc}"
            ) from exc
        self._policy = policy
        self._stdout_buffer = b""
        self._next_sequence = 0
        self._last_timestamp_100ns: int | None = None
        self._closed = False
        try:
            ready = self._read_json_line("Windows thermal-counter initialization")
            ready = _exact_object(
                ready,
                HOST_COUNTER_READY_KEYS,
                "Windows thermal-counter ready record",
            )
            if (
                ready["Schema"] != HOST_COUNTER_SCHEMA
                or ready["CpuName"] != policy.cpu_name
                or ready["Name"] != policy.thermal_zone_name
                or ready["CounterNames"] != list(HOST_COUNTER_NAMES)
            ):
                raise ThermalGuardError(
                    "Windows thermal-counter ready record violates its exact identity"
                )
        except Exception as exc:
            try:
                self.close()
            except ThermalGuardError as close_exc:
                raise ThermalGuardError(f"{exc}; cleanup failed: {close_exc}") from exc
            raise

    def _process_detail(self) -> str:
        returncode = self._process.poll()
        detail = "still running" if returncode is None else f"exited {returncode}"
        if returncode is not None and self._process.stderr is not None:
            try:
                raw = self._process.stderr.read(HOST_COUNTER_LINE_LIMIT + 1)
            except OSError:
                raw = b""
            if raw:
                if len(raw) > HOST_COUNTER_LINE_LIMIT:
                    return detail + " with oversized stderr"
                detail += ": " + raw.decode("utf-8", errors="replace").strip()
        return detail

    def _read_line(self, label: str) -> str:
        if self._process.stdout is None:
            raise ThermalGuardError("Windows thermal-counter stdout is unavailable")
        deadline = time.monotonic() + HOST_COUNTER_TIMEOUT_SECONDS
        while b"\n" not in self._stdout_buffer:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise ThermalGuardError(
                    f"{label} timed out; counter process {self._process_detail()}"
                )
            try:
                readable, _, _ = select.select(
                    [self._process.stdout.fileno()],
                    [],
                    [],
                    remaining,
                )
            except (OSError, ValueError) as exc:
                raise ThermalGuardError(f"{label} cannot poll stdout: {exc}") from exc
            if not readable:
                continue
            try:
                chunk = os.read(self._process.stdout.fileno(), 1024)
            except OSError as exc:
                raise ThermalGuardError(f"{label} cannot read stdout: {exc}") from exc
            if not chunk:
                raise ThermalGuardError(
                    f"{label} reached EOF; counter process {self._process_detail()}"
                )
            self._stdout_buffer += chunk
            if len(self._stdout_buffer) > HOST_COUNTER_LINE_LIMIT:
                raise ThermalGuardError(f"{label} exceeded its line-size bound")
        raw, self._stdout_buffer = self._stdout_buffer.split(b"\n", 1)
        raw = raw.removesuffix(b"\r")
        try:
            return raw.decode("utf-8-sig", errors="strict")
        except UnicodeDecodeError as exc:
            raise ThermalGuardError(f"{label} is not UTF-8: {exc}") from exc

    def _read_json_line(self, label: str) -> Any:
        return _strict_json_loads(self._read_line(label), label)

    def read_millicelsius(self) -> int:
        if self._closed:
            raise ThermalGuardError("Windows thermal counter is closed")
        if self._process.stdin is None:
            raise ThermalGuardError("Windows thermal-counter stdin is unavailable")
        sequence = self._next_sequence
        try:
            self._process.stdin.write(f"{sequence}\n".encode("ascii"))
            self._process.stdin.flush()
        except (BrokenPipeError, OSError) as exc:
            raise ThermalGuardError(
                "cannot request a Windows thermal sample: "
                f"{exc}; counter process {self._process_detail()}"
            ) from exc
        raw = self._read_json_line("Windows thermal-counter sample")
        row = _exact_object(
            raw,
            HOST_COUNTER_SAMPLE_KEYS,
            "Windows thermal-counter sample",
        )
        observed_sequence = _integer(
            row["Sequence"],
            "Windows thermal-counter sample Sequence",
            0,
            2**63 - 1,
        )
        timestamp_100ns = _integer(
            row["Timestamp100nSec"],
            "Windows thermal-counter sample Timestamp100nSec",
            1,
            2**63 - 1,
        )
        if (
            row["Schema"] != HOST_COUNTER_SCHEMA
            or observed_sequence != sequence
        ):
            raise ThermalGuardError(
                "Windows thermal-counter sample violates its schema or sequence"
            )
        if (
            self._last_timestamp_100ns is not None
            and timestamp_100ns <= self._last_timestamp_100ns
        ):
            raise ThermalGuardError(
                "Windows thermal-counter sample timestamp did not advance"
            )
        self._last_timestamp_100ns = timestamp_100ns
        self._next_sequence += 1
        return _parse_host_temperature(
            self._policy,
            {
                "Name": row["Name"],
                "Temperature": row["Temperature"],
                "HighPrecisionTemperature": row["HighPrecisionTemperature"],
                "PercentPassiveLimit": row["PercentPassiveLimit"],
                "ThrottleReasons": row["ThrottleReasons"],
            },
        )

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        errors: list[str] = []
        if self._process.stdin is not None:
            try:
                self._process.stdin.close()
            except OSError as exc:
                errors.append(f"cannot close stdin: {exc}")
        try:
            returncode = self._process.wait(timeout=HOST_COUNTER_TIMEOUT_SECONDS)
        except subprocess.TimeoutExpired:
            errors.append("counter process did not exit after stdin closed")
            self._process.terminate()
            try:
                returncode = self._process.wait(timeout=HOST_COUNTER_TIMEOUT_SECONDS)
            except subprocess.TimeoutExpired:
                self._process.kill()
                returncode = self._process.wait()
        if returncode != 0:
            errors.append(f"counter process exited {returncode}")
        if self._stdout_buffer:
            errors.append("counter process left an unconsumed partial stdout record")
        if self._process.stdout is not None:
            try:
                trailing_stdout = self._process.stdout.read(HOST_COUNTER_LINE_LIMIT + 1)
            except OSError as exc:
                errors.append(f"cannot read trailing stdout: {exc}")
            else:
                if trailing_stdout:
                    errors.append("counter process emitted unexpected trailing stdout")
            self._process.stdout.close()
        if self._process.stderr is not None:
            try:
                trailing_stderr = self._process.stderr.read(
                    HOST_COUNTER_LINE_LIMIT + 1
                )
            except OSError as exc:
                errors.append(f"cannot read trailing stderr: {exc}")
            else:
                if trailing_stderr:
                    detail = trailing_stderr.decode(
                        "utf-8", errors="replace"
                    ).strip()
                    errors.append(f"counter process emitted stderr: {detail}")
            self._process.stderr.close()
        if errors:
            raise ThermalGuardError(
                "persistent Windows thermal-counter cleanup failed: "
                + "; ".join(errors)
            )

    def __enter__(self) -> WindowsThermalCounter:
        return self

    def __exit__(
        self,
        exception_type: type[BaseException] | None,
        exception: BaseException | None,
        traceback: Any,
    ) -> bool:
        try:
            self.close()
        except ThermalGuardError as close_exc:
            if exception is not None:
                raise ThermalGuardError(
                    f"{exception}; thermal-counter cleanup failed: {close_exc}"
                ) from exception
            raise
        return False


class NvmlTemperatureCounter:
    """Persistent exact-UUID NVML temperature counter."""

    def __init__(
        self,
        policy: ThermalPolicy,
        *,
        counter_factory: Callable[..., NvmlMemoryCounter] = NvmlMemoryCounter,
    ) -> None:
        self._counter: NvmlMemoryCounter | None = None
        try:
            counter = counter_factory(
                None,
                device_uuid=policy.gpu_uuid,
                library_name=str(NVIDIA_NVML),
            )
            self._counter = counter
            identity = counter.receipt_identity()["device"]
            if (
                identity["uuid"] != policy.gpu_uuid
                or identity["name"] != policy.gpu_name
            ):
                raise ThermalGuardError(
                    "NVML GPU identity does not match the thermal policy"
                )
        except (DeviceMemoryError, KeyError, TypeError, ThermalGuardError) as exc:
            cleanup_error: DeviceMemoryError | None = None
            if self._counter is not None:
                try:
                    self._counter.close()
                except DeviceMemoryError as close_exc:
                    cleanup_error = close_exc
                self._counter = None
            suffix = (
                f"; cleanup failed: {cleanup_error}"
                if cleanup_error is not None
                else ""
            )
            raise ThermalGuardError(
                "cannot initialize persistent WSL NVML temperature counter: "
                f"{exc}{suffix}"
            ) from exc

    def read_millicelsius(self) -> int:
        if self._counter is None:
            raise ThermalGuardError("persistent WSL NVML temperature counter is closed")
        try:
            return self._counter.read_temperature_millicelsius()
        except DeviceMemoryError as exc:
            raise ThermalGuardError(
                f"cannot read persistent WSL NVML temperature: {exc}"
            ) from exc

    def close(self) -> None:
        if self._counter is None:
            return
        counter = self._counter
        self._counter = None
        try:
            counter.close()
        except DeviceMemoryError as exc:
            raise ThermalGuardError(
                f"cannot close persistent WSL NVML temperature counter: {exc}"
            ) from exc


class PersistentThermalSampler:
    """Lifecycle-bound host and GPU sampler used by the outer supervisor."""

    def __init__(self, policy: ThermalPolicy) -> None:
        self._host = WindowsThermalCounter(policy)
        try:
            self._gpu = NvmlTemperatureCounter(policy)
        except Exception as exc:
            try:
                self._host.close()
            except ThermalGuardError as close_exc:
                raise ThermalGuardError(f"{exc}; cleanup failed: {close_exc}") from exc
            raise
        self._closed = False

    def sample(self) -> ThermalSample:
        if self._closed:
            raise ThermalGuardError("persistent thermal sampler is closed")
        monotonic_seconds = time.monotonic()
        return ThermalSample(
            monotonic_seconds=monotonic_seconds,
            host_millicelsius=self._host.read_millicelsius(),
            gpu_millicelsius=self._gpu.read_millicelsius(),
        )

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        errors: list[str] = []
        for label, counter in (("host", self._host), ("GPU", self._gpu)):
            try:
                counter.close()
            except ThermalGuardError as exc:
                errors.append(f"{label}: {exc}")
        if errors:
            raise ThermalGuardError(
                "persistent thermal sampler cleanup failed: " + "; ".join(errors)
            )

    def __enter__(self) -> PersistentThermalSampler:
        return self

    def __exit__(
        self,
        exception_type: type[BaseException] | None,
        exception: BaseException | None,
        traceback: Any,
    ) -> bool:
        try:
            self.close()
        except ThermalGuardError as close_exc:
            if exception is not None:
                raise ThermalGuardError(
                    f"{exception}; thermal sampler cleanup failed: {close_exc}"
                ) from exception
            raise
        return False

def sample(policy: ThermalPolicy) -> ThermalSample:
    with PersistentThermalSampler(policy) as sampler:
        return sampler.sample()


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
    except OSError as exc:
        error = (
            ThermalTelemetryClosed
            if exc.errno == errno.EPIPE
            else ThermalGuardError
        )
        raise error(
            f"cannot deliver thermal telemetry to the scope controller: {exc}"
        ) from exc
    if written != len(payload):
        raise ThermalGuardError(
            "thermal telemetry pipe accepted only a partial sample"
        )


def _deliver_telemetry_or_settle_child_exit(
    process: subprocess.Popen[Any],
    descriptor: int,
    policy: ThermalPolicy,
    sequence: int,
    value: ThermalSample,
) -> bool:
    try:
        _send_telemetry_sample(descriptor, policy, sequence, value)
    except ThermalTelemetryClosed as exc:
        settlement_seconds = min(
            policy.poll_interval_ms / 1000.0,
            MAX_TELEMETRY_EXIT_SETTLEMENT_SECONDS,
        )
        try:
            process.wait(timeout=settlement_seconds)
        except subprocess.TimeoutExpired:
            raise exc
        except (OSError, subprocess.SubprocessError) as wait_exc:
            raise ThermalGuardError(
                f"{exc}; cannot settle the owned scope exit: {wait_exc}"
            ) from wait_exc
        return False
    return True


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


def _stable_preflight(
    policy: ThermalPolicy,
    sample_reader: Callable[[], ThermalSample],
) -> PreflightResult:
    pacing = policy.pacing
    if pacing is None:
        current = sample_reader()
        _check_limits(current, policy)
        return PreflightResult(
            ending=current,
            sample_count=1,
            peak_host_millicelsius=current.host_millicelsius,
            peak_gpu_millicelsius=current.gpu_millicelsius,
        )

    deadline = time.monotonic() + pacing.timeout_seconds
    stable = 0
    sample_count = 0
    peak_host = 0
    peak_gpu = 0
    while True:
        current = sample_reader()
        sample_count += 1
        peak_host = max(peak_host, current.host_millicelsius)
        peak_gpu = max(peak_gpu, current.gpu_millicelsius)
        _check_limits(current, policy)
        below = (
            current.host_millicelsius <= pacing.host_resume_millicelsius
            and current.gpu_millicelsius <= pacing.gpu_resume_millicelsius
        )
        stable = stable + 1 if below else 0
        if stable >= pacing.resume_stable_samples:
            return PreflightResult(
                ending=current,
                sample_count=sample_count,
                peak_host_millicelsius=peak_host,
                peak_gpu_millicelsius=peak_gpu,
            )
        if time.monotonic() >= deadline:
            raise ThermalGuardError(
                "stable preflight thermal boundary timed out: "
                f"host={current.host_millicelsius}, "
                f"gpu={current.gpu_millicelsius}"
            )
        time.sleep(policy.poll_interval_ms / 1000.0)


def _safe_handoff(
    policy: ThermalPolicy,
    *,
    first: ThermalSample | None = None,
    sample_reader: Callable[[], ThermalSample] | None = None,
) -> HandoffResult:
    deadline = time.monotonic() + policy.handoff_timeout_seconds
    stable = 0
    current = first
    read_sample = sample_reader or (lambda: sample(policy))
    sample_count = 0
    peak_host = 0
    peak_gpu = 0
    limit_failure: ThermalGuardError | None = None
    while True:
        current = current or read_sample()
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
    with PersistentThermalSampler(policy) as sampler:
        preflight = _stable_preflight(policy, sampler.sample)
        return _supervise_with_sampler(policy, command, sampler, preflight)


def _supervise_with_sampler(
    policy: ThermalPolicy,
    command: Sequence[str],
    sampler: PersistentThermalSampler,
    preflight: PreflightResult,
) -> int:
    starting = preflight.ending
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
            _deliver_telemetry_or_settle_child_exit(
                process,
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
    peak_host = preflight.peak_host_millicelsius
    peak_gpu = preflight.peak_gpu_millicelsius
    samples = preflight.sample_count
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
                current = sampler.sample()
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
                    delivered = _deliver_telemetry_or_settle_child_exit(
                        process,
                        telemetry_write_fd,
                        policy,
                        telemetry_sequence,
                        current,
                    )
                except ThermalGuardError as exc:
                    failure = exc
                    break
                if not delivered:
                    break
        if failure is not None:
            _emit("trip", policy, reason=str(failure))
            _terminate(process)
        returncode = process.wait()
        try:
            handoff = _safe_handoff(policy, sample_reader=sampler.sample)
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
