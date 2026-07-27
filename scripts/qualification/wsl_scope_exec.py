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
from pathlib import Path
from typing import Any, Sequence

import wsl_platform


BOUNDARY_ENV = "KILN_WSL2_SCOPE_BOUNDARY"
MEMORY_MAX_ENV = "KILN_WSL2_SCOPE_MEMORY_MAX_BYTES"
PIDS_MAX_ENV = "KILN_WSL2_SCOPE_PIDS_MAX"
CPU_QUOTA_ENV = "KILN_WSL2_SCOPE_CPU_QUOTA_PERCENT"
UNIT_ENV = "KILN_WSL2_SCOPE_UNIT"
HOST_UID_ENV = "KILN_WSL2_SCOPE_HOST_UID"
BOUNDARY_VALUE = "systemd-user-scope-feedback-v1"
SCOPE_EVENT_SCHEMA_V1 = "kiln.wsl2-scope-event.v1"
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


def _reset_scope_failure(unit: str, environment: dict[str, str]) -> None:
    scope_name = f"{unit}.scope"
    try:
        completed = subprocess.run(
            ["/usr/bin/systemctl", "--user", "is-failed", scope_name],
            env=environment,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
            timeout=15.0,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise ScopeExecError(f"scope failed-state probe failed: {exc}") from exc
    stderr = completed.stderr.decode(errors="replace").strip()
    if completed.returncode == 1 and not stderr:
        return
    if completed.returncode != 0:
        raise ScopeExecError(
            f"scope failed-state probe exited {completed.returncode}"
            + (f": {stderr}" if stderr else "")
        )
    _run(
        ["/usr/bin/systemctl", "--user", "reset-failed", scope_name],
        "scope failed-state reset",
        environment=environment,
    )


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
    environment = ensure_user_bus()
    if SCOPE_PAYLOAD.is_symlink() or not SCOPE_PAYLOAD.is_file():
        raise ScopeExecError(f"trusted scope payload is unavailable: {SCOPE_PAYLOAD}")
    unit = f"kiln-wsl-scope-{uuid.uuid4().hex}"
    cgroup = _scope_path(unit)
    handoff = tempfile.TemporaryDirectory(prefix=f"{unit}-")
    handoff_root = Path(handoff.name)
    status_path = handoff_root / "status.json"
    release_path = handoff_root / "release"
    environment.update(
        {
            BOUNDARY_ENV: BOUNDARY_VALUE,
            MEMORY_MAX_ENV: str(args.memory_max_bytes),
            PIDS_MAX_ENV: str(args.pids_max),
            CPU_QUOTA_ENV: str(args.cpu_quota_percent),
            UNIT_ENV: unit,
            HOST_UID_ENV: str(os.getuid()),
        }
    )
    memory_max_property = (
        "infinity" if args.memory_max_bytes == 0 else str(args.memory_max_bytes)
    )
    memory_max_cgroup = (
        "max" if args.memory_max_bytes == 0 else str(args.memory_max_bytes)
    )
    command = [
        "/usr/bin/systemd-run",
        "--user",
        "--scope",
        "--quiet",
        f"--unit={unit}",
        "-p",
        f"MemoryMax={memory_max_property}",
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

    old_handlers = {
        signum: signal.signal(signum, handle_signal)
        for signum in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP)
    }
    try:
        _wait_scope(cgroup, process)
        for filename, expected in (
            ("memory.max", memory_max_cgroup),
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
                "unexpected cpu.max appeared in the WSL2 scope"
            )
        baseline_cpu = _cpu_usage(cgroup)
        controlled_started = time.monotonic()
        _emit(
            "start",
            unit=unit,
            cgroup=str(cgroup),
            containment=os.environ[wsl_platform.NETWORK_ISOLATION_ENV],
            memory_max_bytes=args.memory_max_bytes,
            memory_swap_max_bytes=0,
            pids_max=args.pids_max,
            cpu_quota_percent=args.cpu_quota_percent,
            cpu_controller=(
                "usage-feedback-cgroup-freeze-v1"
                if args.cpu_quota_percent > 0
                else "not_configured"
            ),
            cpu_poll_interval_ms=args.cpu_poll_interval_ms,
            runtime_max_seconds=args.runtime_max_seconds,
        )

        poll_seconds = args.cpu_poll_interval_ms / 1000.0
        burst_usec = (
            max(
                1,
                int(
                    args.cpu_poll_interval_ms
                    * 1000
                    * args.cpu_quota_percent
                    / 100
                ),
            )
            if args.cpu_quota_percent > 0
            else 0
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
            usage = _cpu_usage(cgroup) - baseline_cpu
            allowance = (
                int(elapsed * 1_000_000 * args.cpu_quota_percent / 100)
                if args.cpu_quota_percent > 0
                else None
            )
            cpu_should_freeze = (
                allowance is not None and usage > allowance + burst_usec
            )
            if cpu_should_freeze != frozen:
                _set_frozen(cgroup, cpu_should_freeze)
                frozen = cpu_should_freeze
            peak_memory = max(peak_memory, _integer(cgroup / "memory.peak"))
            peak_pids = max(peak_pids, _integer(cgroup / "pids.peak"))
            last_memory_events = _events(cgroup / "memory.events")
            if status_ready:
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
            if args.cpu_quota_percent > 0:
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
        try:
            _run(
                ["/usr/bin/systemctl", "--user", "stop", f"{unit}.scope"],
                "scope stop",
                environment=environment,
            )
        except ScopeExecError:
            if cgroup.exists() and failure is None:
                failure = ScopeExecError("scope remained after command completion")
        try:
            _reset_scope_failure(unit, environment)
        except ScopeExecError as reset_error:
            failure = (
                reset_error
                if failure is None
                else ScopeExecError(f"{failure}; {reset_error}")
            )
        handoff.cleanup()

    duration = time.monotonic() - controlled_started
    _emit(
        "complete" if failure is None else "failed",
        unit=unit,
        duration_seconds=duration,
        cpu_usage_usec=usage,
        cpu_allowed_usec=int(
            duration * 1_000_000 * args.cpu_quota_percent / 100
        )
        if args.cpu_quota_percent > 0
        else None,
        cpu_quota_percent=args.cpu_quota_percent,
        memory_peak_bytes=peak_memory,
        memory_events=last_memory_events,
        pids_peak=peak_pids,
        scope_removed=not cgroup.exists(),
        child_returncode=child_returncode,
        reason=None if failure is None else str(failure),
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
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args(argv)
    if args.command and args.command[0] == "--":
        args.command = args.command[1:]
    if not args.command:
        parser.error("a command is required after --")
    if (
        args.memory_max_bytes != 0
        and args.memory_max_bytes < 64 * 1024 * 1024
    ):
        parser.error("--memory-max-bytes must be zero or at least 64 MiB")
    if not 1 <= args.pids_max <= 1_000_000:
        parser.error("--pids-max must be in 1..=1000000")
    if not 0 <= args.cpu_quota_percent <= 100:
        parser.error("--cpu-quota-percent must be in 0..=100 (0 disables it)")
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
