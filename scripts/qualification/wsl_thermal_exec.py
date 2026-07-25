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


SCHEMA = "kiln.wsl2-thermal-policy.v1"
POLICY_ENV = "KILN_WSL2_THERMAL_POLICY_SHA256"
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


class ThermalGuardError(RuntimeError):
    """The WSL2 thermal policy or live telemetry failed closed."""


@dataclass(frozen=True)
class ThermalPolicy:
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


@dataclass(frozen=True)
class ThermalSample:
    monotonic_seconds: float
    host_millicelsius: int
    gpu_millicelsius: int


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
    root = _exact_object(raw, POLICY_KEYS, "WSL2 thermal policy")
    if root["schema"] != SCHEMA:
        raise ThermalGuardError(f"WSL2 thermal policy schema must be {SCHEMA!r}")
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
    return ThermalPolicy(
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
) -> ThermalSample:
    deadline = time.monotonic() + policy.handoff_timeout_seconds
    stable = 0
    current = first
    while True:
        current = current or sample(policy)
        _check_limits(current, policy)
        below = (
            current.host_millicelsius <= policy.handoff_host_millicelsius
            and current.gpu_millicelsius <= policy.handoff_gpu_millicelsius
        )
        stable = stable + 1 if below else 0
        if stable >= policy.handoff_stable_samples:
            return current
        if time.monotonic() >= deadline:
            raise ThermalGuardError(
                "safe handoff timed out: "
                f"host={current.host_millicelsius}, gpu={current.gpu_millicelsius}"
            )
        time.sleep(policy.poll_interval_ms / 1000.0)
        current = None


def supervise(policy: ThermalPolicy, command: Sequence[str]) -> int:
    if not command:
        raise ThermalGuardError("a supervised command is required")
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
    try:
        process = subprocess.Popen(list(command), env=child_environment)
    except OSError as exc:
        raise ThermalGuardError(f"cannot launch supervised command: {exc}") from exc

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
        if failure is not None:
            _emit("trip", policy, reason=str(failure))
            _terminate(process)
        returncode = process.wait()
    finally:
        for signum, handler in old_handlers.items():
            signal.signal(signum, handler)

    if interrupted_signal is not None:
        return 128 + interrupted_signal
    if failure is not None:
        raise failure

    ending = _safe_handoff(policy)
    samples += policy.handoff_stable_samples
    peak_host = max(peak_host, ending.host_millicelsius)
    peak_gpu = max(peak_gpu, ending.gpu_millicelsius)
    _emit(
        "complete",
        policy,
        child_returncode=returncode,
        sample_count=samples,
        starting_host_millicelsius=starting.host_millicelsius,
        starting_gpu_millicelsius=starting.gpu_millicelsius,
        peak_host_millicelsius=peak_host,
        peak_gpu_millicelsius=peak_gpu,
        ending_host_millicelsius=ending.host_millicelsius,
        ending_gpu_millicelsius=ending.gpu_millicelsius,
        safe_handoff_stable_samples=policy.handoff_stable_samples,
    )
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
