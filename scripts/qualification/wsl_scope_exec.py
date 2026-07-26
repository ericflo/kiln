#!/usr/bin/env python3
"""Run one WSL2 case in a verified user scope with fail-closed limits."""

from __future__ import annotations

import argparse
import json
import math
import os
import platform
import signal
import stat
import subprocess
import sys
import tempfile
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Sequence

import wsl_platform
import wsl_thermal_exec
from wsl_thermal_exec import (
    PACING_POLICY_ENV,
    POLICY_ENV,
    SHA256_RE,
    TELEMETRY_FD_ENV,
    TELEMETRY_SCHEMA,
)


BOUNDARY_ENV = "KILN_WSL2_SCOPE_BOUNDARY"
MEMORY_MAX_ENV = "KILN_WSL2_SCOPE_MEMORY_MAX_BYTES"
PIDS_MAX_ENV = "KILN_WSL2_SCOPE_PIDS_MAX"
CPU_QUOTA_ENV = "KILN_WSL2_SCOPE_CPU_QUOTA_PERCENT"
UNIT_ENV = "KILN_WSL2_SCOPE_UNIT"
HOST_UID_ENV = "KILN_WSL2_SCOPE_HOST_UID"
PACING_EVENTS_PATH_ENV = "KILN_WSL2_THERMAL_PACING_EVENTS_PATH"
BOUNDARY_VALUE = "systemd-user-scope-feedback-v1"
SCOPE_EVENT_SCHEMA_V1 = "kiln.wsl2-scope-event.v1"
SCOPE_EVENT_SCHEMA_V2 = "kiln.wsl2-scope-event.v2"
THERMAL_TELEMETRY_SOURCE = "outer-supervisor-inherited-pipe-v1"
THERMAL_PACING_EVENT_SCHEMA = "kiln.wsl2-thermal-pacing-event.v1"
THERMAL_TIMEOUT_CLEANUP_GRACE_SECONDS = 75.0
SCOPE_PAYLOAD = Path(__file__).with_name("wsl_scope_payload.py")
SOCKET_UNIT = "dbus.socket"
SERVICE_UNIT = "dbus.service"
SOCKET_TEXT = """[Unit]
Description=Kiln WSL2 User Message Bus

[Socket]
ListenStream=%t/bus
ExecStartPost=-/bin/systemctl --user set-environment DBUS_SESSION_BUS_ADDRESS=unix:path=%t/bus

[Install]
WantedBy=sockets.target
"""
SERVICE_TEXT = """[Unit]
Description=Kiln WSL2 User Message Bus
Requires=dbus.socket

[Service]
ExecStart=@/usr/bin/dbus-daemon dbus-daemon --session --address=systemd: --nofork --nopidfile --systemd-activation --syslog-only
ExecReload=/usr/bin/dbus-send --print-reply --session --type=method_call --dest=org.freedesktop.DBus / org.freedesktop.DBus.ReloadConfig
"""


class ScopeExecError(RuntimeError):
    """The WSL2 user-scope boundary could not be established or verified."""


class ThermalPacingTimeoutError(ScopeExecError):
    """One controller-owned thermal pause exceeded its policy deadline."""


@dataclass
class ThermalPacingState:
    policy: wsl_thermal_exec.ThermalPolicy
    active: bool = False
    pause_started_seconds: float | None = None
    stable_samples: int = 0
    sample_count: int = 0
    pause_count: int = 0
    completed_pause_count: int = 0
    total_pause_seconds: float = 0.0
    longest_pause_seconds: float = 0.0
    peak_host_millicelsius: int = 0
    peak_gpu_millicelsius: int = 0
    ending_host_millicelsius: int | None = None
    ending_gpu_millicelsius: int | None = None

    def observe(
        self,
        value: wsl_thermal_exec.ThermalSample,
        now_seconds: float,
    ) -> bool:
        pacing = self.policy.pacing
        if pacing is None:
            raise ScopeExecError("thermal pacing state lacks a pacing policy")
        try:
            wsl_thermal_exec._check_limits(value, self.policy)
        except wsl_thermal_exec.ThermalGuardError as exc:
            raise ScopeExecError(f"thermal pacing reached a hard limit: {exc}") from exc
        self.sample_count += 1
        self.peak_host_millicelsius = max(
            self.peak_host_millicelsius,
            value.host_millicelsius,
        )
        self.peak_gpu_millicelsius = max(
            self.peak_gpu_millicelsius,
            value.gpu_millicelsius,
        )
        self.ending_host_millicelsius = value.host_millicelsius
        self.ending_gpu_millicelsius = value.gpu_millicelsius
        if not self.active:
            if (
                value.host_millicelsius >= pacing.host_start_millicelsius
                or value.gpu_millicelsius >= pacing.gpu_start_millicelsius
            ):
                self.active = True
                self.pause_started_seconds = now_seconds
                self.stable_samples = 0
                self.pause_count += 1
            return self.active

        if self.pause_started_seconds is None:
            raise ScopeExecError("active thermal pacing has no pause start")
        pause_seconds = now_seconds - self.pause_started_seconds
        if pause_seconds >= pacing.timeout_seconds:
            raise ThermalPacingTimeoutError(
                f"thermal pacing pause exceeded {pacing.timeout_seconds:g} seconds"
            )
        below_resume = (
            value.host_millicelsius <= pacing.host_resume_millicelsius
            and value.gpu_millicelsius <= pacing.gpu_resume_millicelsius
        )
        self.stable_samples = self.stable_samples + 1 if below_resume else 0
        if self.stable_samples >= pacing.resume_stable_samples:
            self.active = False
            self.pause_started_seconds = None
            self.stable_samples = 0
            self.completed_pause_count += 1
            self.total_pause_seconds += pause_seconds
            self.longest_pause_seconds = max(
                self.longest_pause_seconds,
                pause_seconds,
            )
        return self.active

    def record(self, now_seconds: float | None = None) -> dict[str, Any]:
        pacing = self.policy.pacing
        if pacing is None:
            raise ScopeExecError("thermal pacing state lacks a pacing policy")
        total_pause_seconds = self.total_pause_seconds
        longest_pause_seconds = self.longest_pause_seconds
        if self.active:
            if self.pause_started_seconds is None or now_seconds is None:
                raise ScopeExecError(
                    "active thermal pacing evidence requires the current time"
                )
            active_pause_seconds = max(
                0.0,
                now_seconds - self.pause_started_seconds,
            )
            total_pause_seconds += active_pause_seconds
            longest_pause_seconds = max(
                longest_pause_seconds,
                active_pause_seconds,
            )
        return {
            "policy_sha256": self.policy.content_sha256,
            "mode": pacing.mode,
            "active": self.active,
            "sample_count": self.sample_count,
            "pause_count": self.pause_count,
            "completed_pause_count": self.completed_pause_count,
            "total_pause_seconds": total_pause_seconds,
            "longest_pause_seconds": longest_pause_seconds,
            "peak_host_millicelsius": self.peak_host_millicelsius,
            "peak_gpu_millicelsius": self.peak_gpu_millicelsius,
            "ending_host_millicelsius": self.ending_host_millicelsius,
            "ending_gpu_millicelsius": self.ending_gpu_millicelsius,
        }


class ThermalTelemetryReader:
    KEYS = {
        "schema",
        "policy_sha256",
        "sequence",
        "monotonic_seconds",
        "host_millicelsius",
        "gpu_millicelsius",
    }

    def __init__(
        self,
        descriptor: int,
        policy: wsl_thermal_exec.ThermalPolicy,
    ) -> None:
        try:
            metadata = os.fstat(descriptor)
        except OSError as exc:
            raise ScopeExecError(f"cannot inspect thermal telemetry pipe: {exc}") from exc
        if not stat.S_ISFIFO(metadata.st_mode):
            raise ScopeExecError("thermal telemetry descriptor is not a pipe")
        os.set_blocking(descriptor, False)
        self.descriptor = descriptor
        self.policy = policy
        self.buffer = b""
        self.next_sequence = 0
        self.last_monotonic_seconds: float | None = None

    @staticmethod
    def _decode(payload: bytes) -> dict[str, Any]:
        def object_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
            value: dict[str, Any] = {}
            for name, item in pairs:
                if name in value:
                    raise ScopeExecError(
                        f"thermal telemetry repeats field {name!r}"
                    )
                value[name] = item
            return value

        try:
            value = json.loads(payload, object_pairs_hook=object_pairs)
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ScopeExecError(f"thermal telemetry is malformed JSON: {exc}") from exc
        if not isinstance(value, dict) or set(value) != ThermalTelemetryReader.KEYS:
            raise ScopeExecError("thermal telemetry violates its exact schema")
        return value

    def read_available(self) -> list[wsl_thermal_exec.ThermalSample]:
        while True:
            try:
                chunk = os.read(self.descriptor, 65_536)
            except BlockingIOError:
                break
            except OSError as exc:
                raise ScopeExecError(f"cannot read thermal telemetry: {exc}") from exc
            if not chunk:
                raise ScopeExecError("outer thermal telemetry pipe closed early")
            self.buffer += chunk
            if len(self.buffer) > 65_536:
                raise ScopeExecError("thermal telemetry exceeded its buffer bound")

        samples: list[wsl_thermal_exec.ThermalSample] = []
        while b"\n" in self.buffer:
            payload, self.buffer = self.buffer.split(b"\n", 1)
            value = self._decode(payload)
            if (
                value["schema"] != TELEMETRY_SCHEMA
                or value["policy_sha256"] != self.policy.content_sha256
                or value["sequence"] != self.next_sequence
            ):
                raise ScopeExecError(
                    "thermal telemetry schema, policy, or sequence is invalid"
                )
            monotonic = value["monotonic_seconds"]
            host = value["host_millicelsius"]
            gpu = value["gpu_millicelsius"]
            if (
                isinstance(monotonic, bool)
                or not isinstance(monotonic, (int, float))
                or not math.isfinite(monotonic)
                or monotonic < 0
                or isinstance(host, bool)
                or not isinstance(host, int)
                or host < -50_000
                or host > 200_000
                or isinstance(gpu, bool)
                or not isinstance(gpu, int)
                or gpu <= 0
                or gpu > 200_000
            ):
                raise ScopeExecError("thermal telemetry sample values are invalid")
            monotonic_value = float(monotonic)
            if (
                self.last_monotonic_seconds is not None
                and monotonic_value <= self.last_monotonic_seconds
            ):
                raise ScopeExecError("thermal telemetry time did not advance")
            self.next_sequence += 1
            self.last_monotonic_seconds = monotonic_value
            samples.append(
                wsl_thermal_exec.ThermalSample(
                    monotonic_seconds=monotonic_value,
                    host_millicelsius=host,
                    gpu_millicelsius=gpu,
                )
            )
        return samples

    def require_fresh(self, now_seconds: float) -> None:
        maximum_age = max(5.0, self.policy.poll_interval_ms / 1000.0 * 3)
        if (
            self.last_monotonic_seconds is None
            or now_seconds - self.last_monotonic_seconds > maximum_age
        ):
            raise ScopeExecError("outer thermal telemetry is missing or stale")

    def close(self) -> None:
        try:
            os.close(self.descriptor)
        except OSError as exc:
            raise ScopeExecError(f"cannot close thermal telemetry pipe: {exc}") from exc


class ThermalPacingEventWriter:
    def __init__(
        self,
        path: Path,
        policy: wsl_thermal_exec.ThermalPolicy,
    ) -> None:
        descriptor: int | None = None
        try:
            descriptor = os.open(
                path,
                os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_CLOEXEC,
                0o600,
            )
            os.fchmod(descriptor, 0o400)
        except OSError as exc:
            if descriptor is not None:
                os.close(descriptor)
            raise ScopeExecError(
                f"cannot create thermal pacing event stream: {exc}"
            ) from exc
        self.descriptor = descriptor
        self.path = path
        self.policy = policy
        self.sequence = 0

    def transition(
        self,
        *,
        active: bool,
        pause_index: int,
        started_monotonic_seconds: float,
        sample: wsl_thermal_exec.ThermalSample,
    ) -> None:
        duration_seconds = (
            0.0
            if active
            else sample.monotonic_seconds - started_monotonic_seconds
        )
        if (
            pause_index < 1
            or not math.isfinite(started_monotonic_seconds)
            or started_monotonic_seconds < 0
            or not math.isfinite(duration_seconds)
            or duration_seconds < 0
        ):
            raise ScopeExecError("thermal pacing transition values are invalid")
        value = {
            "active": active,
            "duration_seconds": duration_seconds,
            "gpu_millicelsius": sample.gpu_millicelsius,
            "host_millicelsius": sample.host_millicelsius,
            "observed_monotonic_seconds": sample.monotonic_seconds,
            "pause_index": pause_index,
            "policy_sha256": self.policy.content_sha256,
            "schema": THERMAL_PACING_EVENT_SCHEMA,
            "sequence": self.sequence,
            "started_monotonic_seconds": started_monotonic_seconds,
            "transition": "started" if active else "completed",
        }
        payload = (
            json.dumps(
                value,
                allow_nan=False,
                ensure_ascii=True,
                separators=(",", ":"),
                sort_keys=True,
            )
            + "\n"
        ).encode("ascii")
        offset = 0
        try:
            while offset < len(payload):
                written = os.write(self.descriptor, payload[offset:])
                if written <= 0:
                    raise OSError("zero-byte pacing event write")
                offset += written
            os.fsync(self.descriptor)
        except OSError as exc:
            raise ScopeExecError(
                f"cannot append thermal pacing event: {exc}"
            ) from exc
        self.sequence += 1

    def close(self) -> None:
        try:
            os.close(self.descriptor)
        except OSError as exc:
            raise ScopeExecError(
                f"cannot close thermal pacing event stream: {exc}"
            ) from exc


def _observe_thermal_pacing(
    state: ThermalPacingState,
    writer: ThermalPacingEventWriter,
    sample: wsl_thermal_exec.ThermalSample,
    *,
    before_start: Callable[[], None] | None = None,
) -> None:
    was_active = state.active
    previous_started = state.pause_started_seconds
    state.observe(sample, sample.monotonic_seconds)
    if state.active == was_active:
        return
    started = state.pause_started_seconds if state.active else previous_started
    if started is None:
        raise ScopeExecError("thermal pacing transition lacks its start time")
    if state.active and before_start is not None:
        before_start()
    writer.transition(
        active=state.active,
        pause_index=state.pause_count,
        started_monotonic_seconds=started,
        sample=sample,
    )


def _close_thermal_telemetry(
    telemetry: ThermalTelemetryReader | None,
    failure: ScopeExecError | None,
) -> ScopeExecError | None:
    if telemetry is None:
        return failure
    try:
        telemetry.close()
    except ScopeExecError as close_error:
        if failure is None:
            return close_error
        return ScopeExecError(f"{failure}; {close_error}")
    return failure


def _run(
    command: list[str],
    label: str,
    *,
    environment: dict[str, str] | None = None,
    timeout: float = 15.0,
) -> str:
    try:
        completed = subprocess.run(
            command,
            env=environment,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
            timeout=timeout,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise ScopeExecError(f"{label} failed: {exc}") from exc
    stdout = completed.stdout.decode(errors="replace").strip()
    stderr = completed.stderr.decode(errors="replace").strip()
    if completed.returncode != 0:
        raise ScopeExecError(
            f"{label} exited {completed.returncode}"
            + (f": {stderr}" if stderr else "")
        )
    return stdout


def _atomic_runtime_unit(path: Path, content: str) -> None:
    payload = content.encode("ascii")
    try:
        metadata = path.lstat()
    except FileNotFoundError:
        metadata = None
    except OSError as exc:
        raise ScopeExecError(f"cannot inspect runtime unit {path}: {exc}") from exc
    if metadata is not None:
        if (
            not stat.S_ISREG(metadata.st_mode)
            or metadata.st_uid != os.getuid()
            or metadata.st_mode & 0o022
        ):
            raise ScopeExecError(f"runtime unit has unsafe ownership or mode: {path}")
        try:
            existing = path.read_bytes()
        except OSError as exc:
            raise ScopeExecError(f"cannot read runtime unit {path}: {exc}") from exc
        if existing != payload:
            raise ScopeExecError(f"refusing to replace different runtime unit: {path}")
        return
    temporary: Path | None = None
    try:
        descriptor, raw_path = tempfile.mkstemp(
            prefix=f".{path.name}.",
            suffix=".tmp",
            dir=path.parent,
        )
        temporary = Path(raw_path)
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            os.fchmod(handle.fileno(), 0o644)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary, path, follow_symlinks=False)
        except FileExistsError as exc:
            raise ScopeExecError(f"runtime unit appeared concurrently: {path}") from exc
    except OSError as exc:
        raise ScopeExecError(f"cannot publish runtime unit {path}: {exc}") from exc
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def _owned_runtime_directory(path: Path, label: str) -> os.stat_result:
    try:
        metadata = path.lstat()
    except OSError as exc:
        raise ScopeExecError(f"{label} is unavailable: {exc}") from exc
    if (
        not stat.S_ISDIR(metadata.st_mode)
        or metadata.st_uid != os.getuid()
        or metadata.st_mode & 0o022
    ):
        raise ScopeExecError(f"{label} has unsafe type, ownership, or permissions")
    return metadata


def _ensure_runtime_unit_directory(runtime: Path) -> Path:
    _owned_runtime_directory(runtime, "WSL2 user runtime directory")
    systemd_directory = runtime / "systemd"
    _owned_runtime_directory(systemd_directory, "WSL2 user systemd runtime directory")
    unit_directory = systemd_directory / "user"
    try:
        unit_directory.mkdir(mode=0o700)
    except FileExistsError:
        pass
    except OSError as exc:
        raise ScopeExecError(
            f"cannot create WSL2 user runtime unit directory: {exc}"
        ) from exc
    _owned_runtime_directory(unit_directory, "WSL2 user runtime unit directory")
    return unit_directory


def ensure_user_bus() -> dict[str, str]:
    if "microsoft-standard-wsl2" not in platform.release().lower():
        raise ScopeExecError("user-scope execution requires a WSL2 kernel")
    try:
        pid1 = Path("/proc/1/comm").read_text(encoding="ascii").strip()
    except OSError as exc:
        raise ScopeExecError(f"cannot inspect PID 1: {exc}") from exc
    if pid1 != "systemd":
        raise ScopeExecError(f"PID 1 must be systemd, got {pid1!r}")
    for executable in ("/usr/bin/systemctl", "/usr/bin/systemd-run", "/usr/bin/dbus-daemon"):
        if not Path(executable).is_file() or not os.access(executable, os.X_OK):
            raise ScopeExecError(f"required executable is unavailable: {executable}")

    runtime = Path(f"/run/user/{os.getuid()}")
    unit_directory = _ensure_runtime_unit_directory(runtime)

    _atomic_runtime_unit(unit_directory / SOCKET_UNIT, SOCKET_TEXT)
    _atomic_runtime_unit(unit_directory / SERVICE_UNIT, SERVICE_TEXT)
    _run(["/usr/bin/systemctl", "--user", "daemon-reload"], "user daemon-reload")
    _run(
        ["/usr/bin/systemctl", "--user", "start", SOCKET_UNIT],
        "user D-Bus socket activation",
    )
    bus = runtime / "bus"
    try:
        bus_metadata = bus.stat()
    except OSError as exc:
        raise ScopeExecError(f"user D-Bus socket is unavailable: {exc}") from exc
    if not stat.S_ISSOCK(bus_metadata.st_mode) or bus_metadata.st_uid != os.getuid():
        raise ScopeExecError("user D-Bus endpoint is not an owned socket")
    environment = dict(os.environ)
    environment["DBUS_SESSION_BUS_ADDRESS"] = f"unix:path={bus}"
    state = _run(
        ["/usr/bin/systemctl", "--user", "is-system-running"],
        "user manager state",
        environment=environment,
    )
    if state != "running":
        raise ScopeExecError(f"user manager state must be running, got {state!r}")
    return environment


def _read(path: Path) -> str:
    try:
        return path.read_text(encoding="ascii").strip()
    except OSError as exc:
        raise ScopeExecError(f"cannot read {path}: {exc}") from exc


def _write(path: Path, value: str) -> None:
    try:
        path.write_text(value, encoding="ascii")
    except OSError as exc:
        raise ScopeExecError(f"cannot write {path}: {exc}") from exc


def _set_frozen(cgroup: Path, frozen: bool, timeout_seconds: float = 1.0) -> None:
    expected = "1" if frozen else "0"
    _write(cgroup / "cgroup.freeze", expected)
    if _read(cgroup / "cgroup.freeze") != expected:
        raise ScopeExecError("cgroup.freeze did not round-trip")
    deadline = time.monotonic() + timeout_seconds
    while True:
        observed = _events(cgroup / "cgroup.events").get("frozen")
        if observed == int(frozen):
            return
        if observed not in {0, 1}:
            raise ScopeExecError("cgroup.events lacks a valid frozen state")
        if time.monotonic() >= deadline:
            raise ScopeExecError(
                f"cgroup did not {'freeze' if frozen else 'resume'}"
            )
        time.sleep(0.001)


def _events(path: Path) -> dict[str, int]:
    values: dict[str, int] = {}
    for line in _read(path).splitlines():
        fields = line.split()
        if len(fields) != 2 or not fields[1].isdecimal():
            raise ScopeExecError(f"malformed cgroup counter row in {path}: {line!r}")
        values[fields[0]] = int(fields[1])
    return values


def _integer(path: Path) -> int:
    value = _read(path)
    if not value.isdecimal():
        raise ScopeExecError(f"{path} returned non-integer value {value!r}")
    return int(value)


def _cpu_usage(cgroup: Path) -> int:
    values = _events(cgroup / "cpu.stat")
    if "usage_usec" not in values:
        raise ScopeExecError("cpu.stat lacks usage_usec")
    return values["usage_usec"]


def _emit(
    event: str,
    *,
    schema: str = SCOPE_EVENT_SCHEMA_V1,
    **fields: Any,
) -> None:
    print(
        "wsl2-scope: "
        + json.dumps(
            {
                "schema": schema,
                "event": event,
                **fields,
            },
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        ),
        file=sys.stderr,
        flush=True,
    )


def _load_thermal_pacing_policy(
    path: Path | None,
    outer_policy_sha256: str,
) -> wsl_thermal_exec.ThermalPolicy | None:
    pacing_sha256 = os.environ.get(PACING_POLICY_ENV)
    if path is None:
        if pacing_sha256 is not None:
            raise ScopeExecError(
                f"{PACING_POLICY_ENV} is set without --thermal-pacing-policy"
            )
        return None
    if path.is_symlink() or not path.is_file():
        raise ScopeExecError(
            "thermal pacing policy must be a regular non-symlink file"
        )
    try:
        policy = wsl_thermal_exec.load_policy(path)
    except wsl_thermal_exec.ThermalGuardError as exc:
        raise ScopeExecError(f"invalid thermal pacing policy: {exc}") from exc
    if policy.pacing is None:
        raise ScopeExecError("thermal pacing policy must use the v2 pacing schema")
    if (
        policy.content_sha256 != outer_policy_sha256
        or pacing_sha256 != policy.content_sha256
    ):
        raise ScopeExecError(
            "thermal pacing policy does not match both outer policy bindings"
        )
    return policy


def _load_thermal_telemetry(
    policy: wsl_thermal_exec.ThermalPolicy | None,
) -> ThermalTelemetryReader | None:
    raw_descriptor = os.environ.get(TELEMETRY_FD_ENV)
    if policy is None:
        if raw_descriptor is not None:
            raise ScopeExecError(
                f"{TELEMETRY_FD_ENV} is set without a thermal pacing policy"
            )
        return None
    if raw_descriptor is None or not raw_descriptor.isdecimal():
        raise ScopeExecError(
            f"{TELEMETRY_FD_ENV} must bind the inherited telemetry pipe"
        )
    descriptor = int(raw_descriptor)
    if descriptor <= 2:
        raise ScopeExecError("thermal telemetry descriptor must be above stderr")
    return ThermalTelemetryReader(descriptor, policy)


def _thermal_pacing_start_record(
    policy: wsl_thermal_exec.ThermalPolicy,
) -> dict[str, Any]:
    pacing = policy.pacing
    if pacing is None:
        raise ScopeExecError("thermal pacing policy has no pacing configuration")
    return {
        "policy_sha256": policy.content_sha256,
        "mode": pacing.mode,
        "telemetry_source": THERMAL_TELEMETRY_SOURCE,
        "freeze_verification": "cgroup-freeze-and-events-roundtrip-v1",
        "host_start_millicelsius": pacing.host_start_millicelsius,
        "host_resume_millicelsius": pacing.host_resume_millicelsius,
        "gpu_start_millicelsius": pacing.gpu_start_millicelsius,
        "gpu_resume_millicelsius": pacing.gpu_resume_millicelsius,
        "resume_stable_samples": pacing.resume_stable_samples,
        "timeout_seconds": pacing.timeout_seconds,
    }


def _scope_path(unit: str) -> Path:
    uid = os.getuid()
    return Path(
        f"/sys/fs/cgroup/user.slice/user-{uid}.slice/"
        f"user@{uid}.service/app.slice/{unit}.scope"
    )


def _wait_scope(
    path: Path,
    process: subprocess.Popen[Any],
    timeout_seconds: float = 10.0,
) -> None:
    deadline = time.monotonic() + timeout_seconds
    while not path.is_dir():
        if process.poll() is not None:
            raise ScopeExecError(
                f"systemd-run exited {process.returncode} before creating {path}"
            )
        if time.monotonic() >= deadline:
            raise ScopeExecError(f"timed out waiting for user scope {path}")
        time.sleep(0.01)


def _kill_scope(cgroup: Path) -> None:
    if not cgroup.is_dir():
        return
    _write(cgroup / "cgroup.freeze", "0")
    events = _events(cgroup / "cgroup.events")
    if events.get("populated") == 1:
        _write(cgroup / "cgroup.kill", "1")


def _cgroup_process_members(cgroup: Path) -> tuple[tuple[int, str, int], ...]:
    raw_pids = _read(cgroup / "cgroup.procs")
    if not raw_pids:
        return ()
    members: list[tuple[int, str, int]] = []
    for raw_pid in raw_pids.splitlines():
        if not raw_pid.isdecimal() or int(raw_pid) <= 0:
            raise ScopeExecError("cgroup.procs contains an invalid process ID")
        pid = int(raw_pid)
        try:
            stat_line = Path(f"/proc/{pid}/stat").read_text(encoding="ascii")
            fields = stat_line[stat_line.rfind(")") + 2 :].split()
            state = fields[0]
            parent_pid = int(fields[1])
        except FileNotFoundError:
            continue
        except (IndexError, OSError, UnicodeError, ValueError) as exc:
            raise ScopeExecError(
                f"cannot inspect cgroup process {pid}: {exc}"
            ) from exc
        members.append((pid, state, parent_pid))
    return tuple(members)


def _signal_cgroup_member(
    cgroup: Path,
    pid: int,
    signal_number: int,
) -> bool:
    descriptor: int | None = None
    try:
        descriptor = os.pidfd_open(pid)
        members = {
            member_pid: state
            for member_pid, state, _parent_pid in _cgroup_process_members(cgroup)
        }
        if pid not in members or members[pid] == "Z":
            return False
        signal.pidfd_send_signal(descriptor, signal_number)
        return True
    except ProcessLookupError:
        return False
    except OSError as exc:
        raise ScopeExecError(
            f"cannot signal cgroup process {pid}: {exc}"
        ) from exc
    finally:
        if descriptor is not None:
            os.close(descriptor)


def _terminate_scope_leaf_first(
    cgroup: Path,
    grace_seconds: float = THERMAL_TIMEOUT_CLEANUP_GRACE_SECONDS,
) -> bool:
    if not cgroup.is_dir():
        return True
    try:
        _set_frozen(cgroup, False)
    except ScopeExecError:
        if not cgroup.is_dir():
            return True
        raise
    deadline = time.monotonic() + max(0.0, grace_seconds)
    signaled: set[int] = set()
    while True:
        try:
            members = tuple(
                member
                for member in _cgroup_process_members(cgroup)
                if member[1] != "Z"
            )
        except ScopeExecError:
            if not cgroup.is_dir():
                return True
            raise
        if not members:
            return True
        parent_pids = {parent_pid for _pid, _state, parent_pid in members}
        leaves = [
            pid
            for pid, _state, _parent_pid in members
            if pid not in parent_pids and pid not in signaled
        ]
        for pid in leaves:
            try:
                if _signal_cgroup_member(cgroup, pid, signal.SIGINT):
                    signaled.add(pid)
            except ScopeExecError:
                if not cgroup.is_dir():
                    return True
                raise
        if time.monotonic() >= deadline:
            return False
        time.sleep(min(0.05, max(0.0, deadline - time.monotonic())))


def _verify_network_command(command: list[str]) -> None:
    mechanism = os.environ.get(wsl_platform.NETWORK_ISOLATION_ENV)
    if mechanism not in wsl_platform.WSL_CONTAINMENT_MECHANISMS:
        raise ScopeExecError("runner did not bind the accepted WSL2 containment mechanism")
    required_prefix = [
        command[0] if command else "",
        "--user",
        "--map-root-user",
        "--net",
        "--pid",
        "--fork",
        "--kill-child=SIGKILL",
        "--mount",
        "--mount-proc=/proc",
        sys.executable,
        str(Path(__file__).with_name("linux_namespace_exec.py")),
        "--",
    ]
    if (
        len(command) <= len(required_prefix)
        or Path(command[0]).name != "unshare"
        or command[: len(required_prefix)] != required_prefix
    ):
        raise ScopeExecError("scope command is not the required namespace boundary")


def _active_host_builds() -> list[dict[str, Any]]:
    builds: list[dict[str, Any]] = []
    for entry in Path("/proc").iterdir():
        if not entry.name.isdecimal():
            continue
        try:
            comm = (entry / "comm").read_text(encoding="ascii").strip()
            fields = (entry / "stat").read_text(encoding="ascii").split()
        except (FileNotFoundError, PermissionError, ProcessLookupError):
            continue
        except OSError as exc:
            raise ScopeExecError(f"cannot inspect host process {entry.name}: {exc}") from exc
        if comm not in {"cargo", "rustc"} or len(fields) < 3 or fields[2] == "Z":
            continue
        builds.append({"pid": int(entry.name), "comm": comm, "state": fields[2]})
    return sorted(builds, key=lambda value: value["pid"])


def execute(args: argparse.Namespace) -> int:
    _verify_network_command(args.command)
    active_builds = _active_host_builds()
    if active_builds:
        raise ScopeExecError(
            "refusing to overlap host Cargo/rustc processes: "
            + ", ".join(
                f"{value['comm']}[{value['pid']}] state={value['state']}"
                for value in active_builds
            )
        )
    thermal_sha256 = os.environ.get(POLICY_ENV, "")
    if not SHA256_RE.fullmatch(thermal_sha256):
        raise ScopeExecError(f"{POLICY_ENV} does not bind the outer thermal policy")
    thermal_policy = _load_thermal_pacing_policy(
        args.thermal_pacing_policy,
        thermal_sha256,
    )
    thermal_pacing = (
        None if thermal_policy is None else ThermalPacingState(thermal_policy)
    )
    thermal_telemetry = _load_thermal_telemetry(thermal_policy)
    event_schema = (
        SCOPE_EVENT_SCHEMA_V1
        if thermal_pacing is None
        else SCOPE_EVENT_SCHEMA_V2
    )
    environment = ensure_user_bus()
    environment.pop(TELEMETRY_FD_ENV, None)
    environment.pop(PACING_EVENTS_PATH_ENV, None)
    if SCOPE_PAYLOAD.is_symlink() or not SCOPE_PAYLOAD.is_file():
        raise ScopeExecError(f"trusted scope payload is unavailable: {SCOPE_PAYLOAD}")
    unit = f"kiln-wsl-scope-{uuid.uuid4().hex}"
    cgroup = _scope_path(unit)
    handoff = tempfile.TemporaryDirectory(prefix=f"{unit}-")
    handoff_root = Path(handoff.name)
    status_path = handoff_root / "status.json"
    release_path = handoff_root / "release"
    pacing_events_path = handoff_root / "thermal-pacing-events.jsonl"
    pacing_event_writer = (
        None
        if thermal_policy is None
        else ThermalPacingEventWriter(pacing_events_path, thermal_policy)
    )
    environment.update(
        {
            BOUNDARY_ENV: BOUNDARY_VALUE,
            MEMORY_MAX_ENV: str(args.memory_max_bytes),
            PIDS_MAX_ENV: str(args.pids_max),
            CPU_QUOTA_ENV: str(args.cpu_quota_percent),
            UNIT_ENV: unit,
            HOST_UID_ENV: str(os.getuid()),
            **(
                {}
                if pacing_event_writer is None
                else {PACING_EVENTS_PATH_ENV: str(pacing_events_path)}
            ),
        }
    )
    command = [
        "/usr/bin/systemd-run",
        "--user",
        "--scope",
        "--quiet",
        f"--unit={unit}",
        "-p",
        f"MemoryMax={args.memory_max_bytes}",
        "-p",
        "MemorySwapMax=0",
        "-p",
        f"TasksMax={args.pids_max}",
        "--",
        sys.executable,
        str(SCOPE_PAYLOAD),
        "--status-path",
        str(status_path),
        "--release-path",
        str(release_path),
        "--",
        *args.command,
    ]
    try:
        process = subprocess.Popen(command, env=environment)
    except OSError as exc:
        if pacing_event_writer is not None:
            pacing_event_writer.close()
        raise ScopeExecError(f"cannot launch systemd user scope: {exc}") from exc

    interrupted: int | None = None
    failure: ScopeExecError | None = None
    returncode: int | None = None
    frozen = False
    peak_memory = 0
    peak_pids = 0
    last_memory_events: dict[str, int] = {}
    baseline_cpu = 0
    usage = 0
    child_returncode: int | None = None
    controlled_started = time.monotonic()

    def handle_signal(signum: int, _frame: Any) -> None:
        nonlocal interrupted
        interrupted = signum

    def freeze_before_thermal_start() -> None:
        nonlocal frozen
        if not frozen:
            _set_frozen(cgroup, True)
            frozen = True

    old_handlers = {
        signum: signal.signal(signum, handle_signal)
        for signum in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP)
    }
    try:
        _wait_scope(cgroup, process)
        for filename, expected in (
            ("memory.max", str(args.memory_max_bytes)),
            ("memory.swap.max", "0"),
            ("pids.max", str(args.pids_max)),
        ):
            observed = _read(cgroup / filename)
            if observed != expected:
                raise ScopeExecError(
                    f"{filename} round-trip mismatch: {observed!r} != {expected!r}"
                )
        _write(cgroup / "memory.oom.group", "1")
        if _read(cgroup / "memory.oom.group") != "1":
            raise ScopeExecError("memory.oom.group did not round-trip")
        if (cgroup / "cpu.max").exists():
            raise ScopeExecError(
                "unexpected cpu.max appeared; feedback policy requires absent CPU delegation"
            )
        baseline_cpu = _cpu_usage(cgroup)
        controlled_started = time.monotonic()
        if thermal_pacing is not None:
            if thermal_telemetry is None:
                raise ScopeExecError("thermal pacing lacks outer telemetry")
            initial_deadline = time.monotonic() + max(
                5.0,
                thermal_pacing.policy.poll_interval_ms / 1000.0 * 3,
            )
            initial_samples: list[wsl_thermal_exec.ThermalSample] = []
            while not initial_samples:
                initial_samples = thermal_telemetry.read_available()
                if not initial_samples and time.monotonic() >= initial_deadline:
                    raise ScopeExecError("timed out waiting for outer thermal telemetry")
                if not initial_samples:
                    time.sleep(0.005)
            for current_thermal in initial_samples:
                if pacing_event_writer is None:
                    raise ScopeExecError("thermal pacing event stream is unavailable")
                _observe_thermal_pacing(
                    thermal_pacing,
                    pacing_event_writer,
                    current_thermal,
                    before_start=freeze_before_thermal_start,
                )
            thermal_telemetry.require_fresh(time.monotonic())
        _emit(
            "start",
            schema=event_schema,
            unit=unit,
            cgroup=str(cgroup),
            containment=os.environ[wsl_platform.NETWORK_ISOLATION_ENV],
            memory_max_bytes=args.memory_max_bytes,
            memory_swap_max_bytes=0,
            pids_max=args.pids_max,
            cpu_quota_percent=args.cpu_quota_percent,
            cpu_controller="usage-feedback-cgroup-freeze-v1",
            cpu_poll_interval_ms=args.cpu_poll_interval_ms,
            runtime_max_seconds=args.runtime_max_seconds,
            thermal_policy_sha256=thermal_sha256,
            **(
                {}
                if thermal_policy is None
                else {
                    "thermal_pacing": _thermal_pacing_start_record(
                        thermal_policy
                    )
                }
            ),
        )

        poll_seconds = args.cpu_poll_interval_ms / 1000.0
        burst_usec = max(
            1,
            int(args.cpu_poll_interval_ms * 1000 * args.cpu_quota_percent / 100),
        )
        while True:
            if process.poll() is not None:
                raise ScopeExecError(
                    f"scope payload exited {process.returncode} before status handoff"
                )
            now = time.monotonic()
            elapsed = now - controlled_started
            if interrupted is not None:
                failure = ScopeExecError(f"received signal {interrupted}")
                break
            if elapsed >= args.runtime_max_seconds:
                failure = ScopeExecError(
                    f"runtime exceeded {args.runtime_max_seconds:g} seconds"
                )
                break
            status_ready = status_path.exists()
            if thermal_pacing is not None:
                if thermal_telemetry is None:
                    raise ScopeExecError("thermal pacing lost outer telemetry")
                for current_thermal in thermal_telemetry.read_available():
                    if pacing_event_writer is None:
                        raise ScopeExecError(
                            "thermal pacing event stream is unavailable"
                        )
                    _observe_thermal_pacing(
                        thermal_pacing,
                        pacing_event_writer,
                        current_thermal,
                        before_start=freeze_before_thermal_start,
                    )
                thermal_telemetry.require_fresh(time.monotonic())
            usage = _cpu_usage(cgroup) - baseline_cpu
            allowance = int(elapsed * 1_000_000 * args.cpu_quota_percent / 100)
            cpu_should_freeze = usage > allowance + burst_usec
            should_freeze = cpu_should_freeze or (
                thermal_pacing is not None and thermal_pacing.active
            )
            if should_freeze != frozen:
                _set_frozen(cgroup, should_freeze)
                frozen = should_freeze
            peak_memory = max(peak_memory, _integer(cgroup / "memory.peak"))
            peak_pids = max(peak_pids, _integer(cgroup / "pids.peak"))
            last_memory_events = _events(cgroup / "memory.events")
            if status_ready and not (
                thermal_pacing is not None and thermal_pacing.active
            ):
                break
            time.sleep(poll_seconds)

        if failure is not None:
            _kill_scope(cgroup)
        if failure is None:
            try:
                status_value = json.loads(status_path.read_bytes())
            except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
                raise ScopeExecError(f"scope payload status is malformed: {exc}") from exc
            if (
                not isinstance(status_value, dict)
                or set(status_value) != {"schema", "returncode"}
                or status_value["schema"]
                != "kiln.wsl2-scope-payload-status.v1"
                or not isinstance(status_value["returncode"], int)
                or isinstance(status_value["returncode"], bool)
            ):
                raise ScopeExecError("scope payload status violates its exact schema")
            child_returncode = status_value["returncode"]
        if frozen and cgroup.is_dir():
            _set_frozen(cgroup, False)
            frozen = False
        if failure is None:
            settlement_deadline = time.monotonic() + 5.0
            while True:
                elapsed = time.monotonic() - controlled_started
                usage = _cpu_usage(cgroup) - baseline_cpu
                allowance = int(
                    elapsed * 1_000_000 * args.cpu_quota_percent / 100
                )
                peak_memory = max(peak_memory, _integer(cgroup / "memory.peak"))
                peak_pids = max(peak_pids, _integer(cgroup / "pids.peak"))
                last_memory_events = _events(cgroup / "memory.events")
                if usage <= allowance:
                    break
                if time.monotonic() >= settlement_deadline:
                    raise ScopeExecError(
                        "CPU accounting did not settle below the aggregate quota"
                    )
                time.sleep(poll_seconds)
            try:
                descriptor = os.open(
                    release_path,
                    os.O_WRONLY | os.O_CREAT | os.O_EXCL,
                    0o600,
                )
                os.close(descriptor)
            except OSError as exc:
                raise ScopeExecError(f"cannot release scope payload: {exc}") from exc
        returncode = process.wait(timeout=20)
    except (OSError, subprocess.TimeoutExpired, ScopeExecError) as exc:
        failure = exc if isinstance(exc, ScopeExecError) else ScopeExecError(str(exc))
        try:
            settled = (
                _terminate_scope_leaf_first(cgroup)
                if isinstance(failure, ThermalPacingTimeoutError)
                else False
            )
            if not settled:
                _kill_scope(cgroup)
        except ScopeExecError as cleanup_exc:
            failure = ScopeExecError(f"{failure}; scope kill failed: {cleanup_exc}")
        try:
            process.wait(timeout=20)
        except (subprocess.TimeoutExpired, OSError):
            process.kill()
            process.wait()
    finally:
        for signum, handler in old_handlers.items():
            signal.signal(signum, handler)
        failure = _close_thermal_telemetry(thermal_telemetry, failure)
        if pacing_event_writer is not None:
            try:
                pacing_event_writer.close()
            except ScopeExecError as close_error:
                failure = (
                    close_error
                    if failure is None
                    else ScopeExecError(f"{failure}; {close_error}")
                )
        try:
            _run(
                ["/usr/bin/systemctl", "--user", "stop", f"{unit}.scope"],
                "scope stop",
                environment=environment,
            )
        except ScopeExecError:
            if cgroup.exists() and failure is None:
                failure = ScopeExecError("scope remained after command completion")
        handoff.cleanup()

    duration = time.monotonic() - controlled_started
    _emit(
        "complete" if failure is None else "failed",
        schema=event_schema,
        unit=unit,
        duration_seconds=duration,
        cpu_usage_usec=usage,
        cpu_allowed_usec=int(
            duration * 1_000_000 * args.cpu_quota_percent / 100
        ),
        cpu_quota_percent=args.cpu_quota_percent,
        memory_peak_bytes=peak_memory,
        memory_events=last_memory_events,
        pids_peak=peak_pids,
        scope_removed=not cgroup.exists(),
        child_returncode=child_returncode,
        reason=None if failure is None else str(failure),
        **(
            {}
            if thermal_pacing is None
            else {"thermal_pacing": thermal_pacing.record(time.monotonic())}
        ),
    )
    if interrupted is not None:
        return 128 + interrupted
    if failure is not None:
        raise failure
    if child_returncode is None:
        raise ScopeExecError("scope command return code was not collected")
    return (
        child_returncode
        if child_returncode >= 0
        else 128 - child_returncode
    )


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--memory-max-bytes", type=int, required=True)
    parser.add_argument("--pids-max", type=int, default=512)
    parser.add_argument("--cpu-quota-percent", type=int, required=True)
    parser.add_argument("--cpu-poll-interval-ms", type=int, default=5)
    parser.add_argument("--runtime-max-seconds", type=float, required=True)
    parser.add_argument("--thermal-pacing-policy", type=Path)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args(argv)
    if args.command and args.command[0] == "--":
        args.command = args.command[1:]
    if not args.command:
        parser.error("a command is required after --")
    if args.memory_max_bytes < 64 * 1024 * 1024:
        parser.error("--memory-max-bytes must be at least 64 MiB")
    if not 1 <= args.pids_max <= 1_000_000:
        parser.error("--pids-max must be in 1..=1000000")
    if not 1 <= args.cpu_quota_percent <= 100:
        parser.error("--cpu-quota-percent must be in 1..=100")
    if not 1 <= args.cpu_poll_interval_ms <= 1000:
        parser.error("--cpu-poll-interval-ms must be in 1..=1000")
    if (
        not math.isfinite(args.runtime_max_seconds)
        or not 1 <= args.runtime_max_seconds <= 86_400
    ):
        parser.error("--runtime-max-seconds must be in 1..=86400")
    return args


def main(argv: Sequence[str] | None = None) -> int:
    try:
        return execute(parse_args(argv))
    except ScopeExecError as exc:
        print(f"error: WSL2 scope boundary: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
