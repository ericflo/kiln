#!/usr/bin/env python3
"""Collect fail-closed macOS and Metal platform provenance."""

from __future__ import annotations

import errno
import fcntl
import hashlib
import json
import os
import plistlib
import re
import select
import shutil
import socket
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
NETWORK_ISOLATION_ENV = "KILN_QUALIFICATION_NETWORK_ISOLATION"
MACOS_CONTAINMENT_MECHANISMS = {"macos-sandbox-loopback-only-v1"}
CAPABILITY_KEYS = {
    "apple_hardware_identity",
    "metal_runtime_identity",
    "toolchain_provenance",
    "metal_compiler",
    "filesystem_semantics",
    "network_containment",
    "process_containment",
    "unified_memory_accounting",
    "memory_pressure",
    "thermal_pressure",
    "host_temperature",
    "gpu_temperature",
}
F_FULLFSYNC = 51


class PlatformProbeError(RuntimeError):
    """A macOS platform probe did not establish its claimed capability."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return f"sha256:{digest.hexdigest()}"


def _file_identity(path: Path) -> dict[str, Any]:
    try:
        resolved = path.resolve(strict=True)
    except OSError as exc:
        raise PlatformProbeError(f"cannot resolve {path}: {exc}") from exc
    if not resolved.is_file():
        raise PlatformProbeError(f"{resolved} is not a regular file")
    return {
        "path": str(path),
        "resolved_path": str(resolved),
        "bytes": resolved.stat().st_size,
        "sha256": _sha256(resolved),
    }


def _result(
    probe_id: str,
    *,
    required: bool,
    passed: bool,
    started: float,
    detail: str,
) -> dict[str, Any]:
    return {
        "id": probe_id,
        "required": required,
        "status": "passed" if passed else ("failed" if required else "skipped"),
        "duration_seconds": time.monotonic() - started,
        "metrics": [],
        "details": detail[:2048],
    }


def _command(
    probe_id: str,
    argv: list[str],
    raw: dict[str, Any],
    *,
    timeout: float = 30.0,
    retain_output: bool = True,
) -> str:
    try:
        completed = subprocess.run(
            argv,
            cwd=ROOT,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=False,
            timeout=timeout,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raw[probe_id] = {
            "argv": argv,
            "returncode": None,
            "stdout": "",
            "stderr": str(exc),
        }
        raise PlatformProbeError(f"{probe_id} could not execute: {exc}") from exc
    raw[probe_id] = {
        "argv": argv,
        "returncode": completed.returncode,
        "stdout": completed.stdout if retain_output else "<parsed and redacted>",
        "stderr": completed.stderr,
    }
    if completed.returncode != 0:
        detail = completed.stderr.strip()[-500:]
        raise PlatformProbeError(
            f"{probe_id} exited {completed.returncode}"
            + (f": {detail}" if detail else "")
        )
    return completed.stdout


def _positive_integer(value: Any, label: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise PlatformProbeError(f"{label} is not a positive integer")
    return value


def parse_hardware_profile(text: str) -> dict[str, Any]:
    try:
        document = json.loads(text)
        values = document["SPHardwareDataType"]
        hardware = values[0]
    except (json.JSONDecodeError, KeyError, IndexError, TypeError) as exc:
        raise PlatformProbeError(f"invalid SPHardwareDataType JSON: {exc}") from exc
    if not isinstance(hardware, dict):
        raise PlatformProbeError("SPHardwareDataType entry is not an object")
    required = ("machine_name", "machine_model", "chip_type", "physical_memory")
    result: dict[str, str] = {}
    for key in required:
        value = hardware.get(key)
        if not isinstance(value, str) or not value.strip():
            raise PlatformProbeError(f"SPHardwareDataType omitted {key}")
        result[key] = value.strip()
    return result


def parse_memory_size(text: str) -> int:
    match = re.fullmatch(r"\s*(\d+)\s+(GB|MB)\s*", text)
    if match is None:
        raise PlatformProbeError(f"unsupported physical-memory value {text!r}")
    multiplier = 1024**3 if match.group(2) == "GB" else 1024**2
    return int(match.group(1)) * multiplier


def parse_vm_stat(text: str) -> dict[str, int]:
    header = re.search(r"page size of (\d+) bytes", text)
    if header is None:
        raise PlatformProbeError("vm_stat omitted its page size")
    page_size = int(header.group(1))
    if page_size <= 0:
        raise PlatformProbeError("vm_stat reported an invalid page size")
    pages: dict[str, int] = {}
    for line in text.splitlines()[1:]:
        match = re.fullmatch(r'("?[^":]+"?):\s+(\d+)\.', line.strip())
        if match is not None:
            pages[match.group(1).strip('"')] = int(match.group(2))
    required = {
        "Pages free",
        "Pages active",
        "Pages inactive",
        "Pages speculative",
        "Pages wired down",
        "Pages occupied by compressor",
        "Pageins",
        "Pageouts",
        "Swapins",
        "Swapouts",
    }
    if not required <= pages.keys():
        raise PlatformProbeError(
            "vm_stat omitted " + ", ".join(sorted(required - pages.keys()))
        )
    return {
        "page_size_bytes": page_size,
        **{
            re.sub(r"[^a-z0-9]+", "_", key.lower()).strip("_"): pages[key]
            for key in sorted(required)
        },
    }


def parse_memory_pressure(text: str) -> dict[str, int]:
    total = re.search(r"system has (\d+) \((\d+) pages with a page size of (\d+)\)", text)
    free = re.search(r"System-wide memory free percentage:\s*(\d+)%", text)
    if total is None or free is None:
        raise PlatformProbeError("memory_pressure output omitted capacity or free percentage")
    total_bytes, pages, page_size = (int(value) for value in total.groups())
    free_percent = int(free.group(1))
    if total_bytes != pages * page_size or not 0 <= free_percent <= 100:
        raise PlatformProbeError("memory_pressure reported inconsistent values")
    return {
        "total_bytes": total_bytes,
        "page_count": pages,
        "page_size_bytes": page_size,
        "free_percent": free_percent,
    }


def parse_swapusage(text: str) -> dict[str, int | bool]:
    match = re.search(
        r"total = ([0-9.]+)M\s+used = ([0-9.]+)M\s+free = ([0-9.]+)M"
        r"\s+\((encrypted|unencrypted)\)",
        text,
    )
    if match is None:
        raise PlatformProbeError("vm.swapusage output is malformed")
    values = [round(float(value) * 1024**2) for value in match.groups()[:3]]
    total, used, free = values
    if total < 0 or used < 0 or free < 0 or abs(total - used - free) > 1024**2:
        raise PlatformProbeError("vm.swapusage reported inconsistent values")
    return {
        "total_bytes": total,
        "used_bytes": used,
        "free_bytes": free,
        "encrypted": match.group(4) == "encrypted",
    }


def parse_thermal_pressure(text: str) -> dict[str, str]:
    labels = {
        "thermal_warning": "thermal warning level",
        "performance_warning": "performance warning level",
        "cpu_power_status": "CPU power status",
    }
    result: dict[str, str] = {}
    for key, label in labels.items():
        match = re.search(
            rf"(?:No {re.escape(label)} has been recorded|{re.escape(label)}:\s*(.+))",
            text,
            re.IGNORECASE,
        )
        if match is None:
            raise PlatformProbeError(f"pmset omitted {label}")
        result[key] = "not_recorded" if match.group(1) is None else match.group(1).strip()
    return result


def parse_metal_runtime(text: str) -> dict[str, Any]:
    try:
        value = json.loads(text)
    except json.JSONDecodeError as exc:
        raise PlatformProbeError(f"Metal runtime emitted invalid JSON: {exc}") from exc
    required = {
        "name",
        "has_unified_memory",
        "max_buffer_length_bytes",
        "recommended_max_working_set_bytes",
        "current_allocated_bytes",
    }
    if not isinstance(value, dict) or set(value) != required:
        raise PlatformProbeError("Metal runtime identity has an unexpected shape")
    if not isinstance(value["name"], str) or not value["name"]:
        raise PlatformProbeError("Metal runtime device name is empty")
    if value["has_unified_memory"] is not True:
        raise PlatformProbeError("Metal runtime device is not unified-memory")
    for key in (
        "max_buffer_length_bytes",
        "recommended_max_working_set_bytes",
    ):
        _positive_integer(value[key], f"Metal runtime {key}")
    current = value["current_allocated_bytes"]
    if not isinstance(current, int) or isinstance(current, bool) or current < 0:
        raise PlatformProbeError("Metal runtime current_allocated_bytes is invalid")
    return value


def _hardware_probe(
    selected_device: dict[str, Any],
    raw: dict[str, Any],
) -> dict[str, Any]:
    profiler = shutil.which("system_profiler") or "/usr/sbin/system_profiler"
    sysctl = shutil.which("sysctl") or "/usr/sbin/sysctl"
    text = _command(
        "macos-hardware-profile",
        [profiler, "SPHardwareDataType", "-json"],
        raw,
        timeout=90.0,
        retain_output=False,
    )
    profile = parse_hardware_profile(text)
    memory_bytes = int(
        _command(
            "macos-hw-memsize",
            [sysctl, "-n", "hw.memsize"],
            raw,
        ).strip()
    )
    cpu_brand = _command(
        "macos-cpu-brand",
        [sysctl, "-n", "machdep.cpu.brand_string"],
        raw,
    ).strip()
    os_build = _command(
        "macos-kernel-build",
        [sysctl, "-n", "kern.osversion"],
        raw,
    ).strip()
    profile_memory = parse_memory_size(profile["physical_memory"])
    selected_memory = selected_device.get("memory_bytes")
    if (
        profile["chip_type"] != selected_device.get("name")
        or cpu_brand != profile["chip_type"]
        or memory_bytes != profile_memory
        or memory_bytes != selected_memory
    ):
        raise PlatformProbeError(
            "system_profiler, sysctl, and selected Metal identity disagree"
        )
    compute_units = _positive_integer(
        selected_device.get("compute_units"),
        "selected Metal GPU-core count",
    )
    result = {
        "machine_name": profile["machine_name"],
        "machine_model": profile["machine_model"],
        "chip_type": profile["chip_type"],
        "cpu_brand": cpu_brand,
        "gpu_core_count": compute_units,
        "physical_memory_bytes": memory_bytes,
        "kernel_build": os_build,
    }
    raw["macos_hardware_identity"] = result
    return result


def _metal_runtime_probe(
    selected_device: dict[str, Any],
    raw: dict[str, Any],
) -> dict[str, Any]:
    xcrun = shutil.which("xcrun") or "/usr/bin/xcrun"
    source = """
import Foundation
import Metal

guard let device = MTLCreateSystemDefaultDevice() else {
    exit(2)
}
let value: [String: Any] = [
    "name": device.name,
    "has_unified_memory": device.hasUnifiedMemory,
    "max_buffer_length_bytes": Int(device.maxBufferLength),
    "recommended_max_working_set_bytes": Int(device.recommendedMaxWorkingSetSize),
    "current_allocated_bytes": Int(device.currentAllocatedSize),
]
let data = try JSONSerialization.data(withJSONObject: value, options: [.sortedKeys])
FileHandle.standardOutput.write(data)
"""
    probe_root = ROOT / ".qualification" / "macos-platform-probes"
    probe_root.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="runtime-", dir=probe_root) as temporary:
        source_path = Path(temporary) / "device.swift"
        source_path.write_text(source)
        text = _command(
            "macos-metal-runtime",
            [xcrun, "swift", str(source_path)],
            raw,
            timeout=120.0,
        )
    runtime = parse_metal_runtime(text)
    if runtime["name"] != selected_device.get("name"):
        raise PlatformProbeError(
            f"Metal runtime device {runtime['name']!r} disagrees with "
            f"system_profiler {selected_device.get('name')!r}"
        )
    raw["metal_runtime_identity"] = runtime
    return runtime


def _toolchain_probe(raw: dict[str, Any]) -> dict[str, Any]:
    xcode_select = shutil.which("xcode-select") or "/usr/bin/xcode-select"
    xcodebuild = shutil.which("xcodebuild") or "/usr/bin/xcodebuild"
    xcrun = shutil.which("xcrun") or "/usr/bin/xcrun"
    pkgutil = shutil.which("pkgutil") or "/usr/sbin/pkgutil"
    developer_dir = _command(
        "macos-developer-directory", [xcode_select, "-p"], raw
    ).strip()
    xcode_version = _command(
        "macos-xcode-version", [xcodebuild, "-version"], raw
    ).strip()
    clt_package = _command(
        "macos-clt-package",
        [pkgutil, "--pkg-info=com.apple.pkg.CLTools_Executables"],
        raw,
    ).strip()
    sdk_path = _command(
        "macos-sdk-path", [xcrun, "--sdk", "macosx", "--show-sdk-path"], raw
    ).strip()
    sdk_version = _command(
        "macos-sdk-version",
        [xcrun, "--sdk", "macosx", "--show-sdk-version"],
        raw,
    ).strip()
    sdk_build = _command(
        "macos-sdk-build",
        [xcrun, "--sdk", "macosx", "--show-sdk-build-version"],
        raw,
    ).strip()
    metal_path = Path(
        _command("macos-metal-path", [xcrun, "--find", "metal"], raw).strip()
    )
    metallib_path = Path(
        _command("macos-metallib-path", [xcrun, "--find", "metallib"], raw).strip()
    )
    swift_path = Path(
        _command("macos-swift-path", [xcrun, "--find", "swift"], raw).strip()
    )
    if not Path(developer_dir).is_dir() or not Path(sdk_path).is_dir():
        raise PlatformProbeError("developer directory or macOS SDK path is unavailable")
    result = {
        "developer_directory": developer_dir,
        "xcode_version": xcode_version,
        "command_line_tools_package": clt_package,
        "sdk_path": sdk_path,
        "sdk_version": sdk_version,
        "sdk_build": sdk_build,
        "metal": _file_identity(metal_path),
        "metallib": _file_identity(metallib_path),
        "swift": _file_identity(swift_path),
    }
    raw["macos_toolchain_provenance"] = result
    return result


def _metal_compiler_probe(raw: dict[str, Any]) -> dict[str, Any]:
    xcrun = shutil.which("xcrun") or "/usr/bin/xcrun"
    probe_root = ROOT / ".qualification" / "macos-platform-probes"
    probe_root.mkdir(parents=True, exist_ok=True)
    source = """
#include <metal_stdlib>
using namespace metal;

kernel void kiln_platform_probe(
    device uint *values [[buffer(0)]],
    uint index [[thread_position_in_grid]]
) {
    values[index] += 1;
}
"""
    with tempfile.TemporaryDirectory(prefix="compiler-", dir=probe_root) as temporary:
        root = Path(temporary)
        source_path = root / "probe.metal"
        air_path = root / "probe.air"
        library_path = root / "probe.metallib"
        source_path.write_text(source)
        _command(
            "macos-metal-compile",
            [xcrun, "metal", "-c", str(source_path), "-o", str(air_path)],
            raw,
            timeout=120.0,
        )
        _command(
            "macos-metallib-link",
            [xcrun, "metallib", str(air_path), "-o", str(library_path)],
            raw,
            timeout=120.0,
        )
        if air_path.stat().st_size <= 0 or library_path.stat().st_size <= 0:
            raise PlatformProbeError("Metal compiler emitted an empty artifact")
        result = {
            "source_sha256": _sha256(source_path),
            "air_bytes": air_path.stat().st_size,
            "air_sha256": _sha256(air_path),
            "metallib_bytes": library_path.stat().st_size,
            "metallib_sha256": _sha256(library_path),
        }
    raw["metal_compiler_probe"] = result
    return result


def _filesystem_probe(raw: dict[str, Any]) -> dict[str, Any]:
    df = shutil.which("df") or "/bin/df"
    diskutil = shutil.which("diskutil") or "/usr/sbin/diskutil"
    lines = _command(
        "macos-filesystem-df",
        [df, "-Pk", str(ROOT)],
        raw,
    ).splitlines()
    if len(lines) != 2:
        raise PlatformProbeError("df did not return exactly one filesystem row")
    fields = lines[1].split()
    if len(fields) < 6 or not fields[0].startswith("/dev/"):
        raise PlatformProbeError("qualification root is not on a local device filesystem")
    source = fields[0]
    plist_text = _command(
        "macos-filesystem-diskutil",
        [diskutil, "info", "-plist", source],
        raw,
        retain_output=False,
    )
    try:
        disk = plistlib.loads(plist_text.encode())
    except (plistlib.InvalidFileException, ValueError) as exc:
        raise PlatformProbeError(f"diskutil returned invalid plist: {exc}") from exc
    fstype = disk.get("FilesystemType")
    mount_point = disk.get("MountPoint")
    if (
        not isinstance(fstype, str)
        or not fstype
        or not isinstance(mount_point, str)
        or not mount_point
        or disk.get("Writable") is not True
    ):
        raise PlatformProbeError("diskutil did not report a writable filesystem")

    probe_root = ROOT / ".qualification" / "macos-platform-probes"
    probe_root.mkdir(parents=True, exist_ok=True)
    directory = Path(tempfile.mkdtemp(prefix="filesystem-", dir=probe_root))
    try:
        original = directory / "Case"
        different_case = directory / "case"
        replacement = directory / "replacement"
        original.write_bytes(b"original")
        with original.open("rb") as handle:
            fcntl.fcntl(handle.fileno(), F_FULLFSYNC)
        original_inode = original.stat().st_ino
        os.link(original, directory / "hardlink")
        os.symlink("Case", directory / "symlink")
        try:
            with different_case.open("xb") as handle:
                handle.write(b"different")
            case_sensitive = different_case.stat().st_ino != original_inode
        except FileExistsError:
            case_sensitive = False
        if not case_sensitive and different_case.stat().st_ino != original_inode:
            raise PlatformProbeError("case-insensitive lookup did not preserve inode identity")
        replacement.write_bytes(b"replacement")
        with replacement.open("rb") as handle:
            fcntl.fcntl(handle.fileno(), F_FULLFSYNC)
        os.replace(replacement, original)
        directory_fd = os.open(directory, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
        if original.read_bytes() != b"replacement":
            raise PlatformProbeError("atomic replacement did not publish new bytes")
        if (directory / "hardlink").read_bytes() != b"original":
            raise PlatformProbeError("atomic replacement mutated the linked old inode")
        if (directory / "symlink").resolve() != original.resolve():
            raise PlatformProbeError("relative symlink did not resolve")
        if original.stat().st_ino == original_inode:
            raise PlatformProbeError("atomic replacement retained the old inode")
        if case_sensitive and different_case.read_bytes() != b"different":
            raise PlatformProbeError("case-sensitive peer changed unexpectedly")
    finally:
        shutil.rmtree(directory, ignore_errors=True)
    result = {
        "root": str(ROOT),
        "source": source,
        "fstype": fstype,
        "mount_point": mount_point,
        "atomic_replace": True,
        "full_file_sync": True,
        "directory_fsync": True,
        "hardlink": True,
        "symlink": True,
        "case_sensitive": case_sensitive,
    }
    raw["filesystem_semantics"] = result
    return result


def _memory_probe(
    selected_device: dict[str, Any],
    raw: dict[str, Any],
) -> dict[str, Any]:
    vm_stat = shutil.which("vm_stat") or "/usr/bin/vm_stat"
    memory_pressure = shutil.which("memory_pressure") or "/usr/bin/memory_pressure"
    sysctl = shutil.which("sysctl") or "/usr/sbin/sysctl"
    vm = parse_vm_stat(_command("macos-vm-stat", [vm_stat], raw))
    pressure = parse_memory_pressure(
        _command("macos-memory-pressure", [memory_pressure, "-Q"], raw)
    )
    swap = parse_swapusage(
        _command("macos-swap-usage", [sysctl, "-n", "vm.swapusage"], raw)
    )
    if pressure["total_bytes"] != selected_device.get("memory_bytes"):
        raise PlatformProbeError(
            "memory_pressure total disagrees with selected unified-memory capacity"
        )
    result = {
        "total_bytes": pressure["total_bytes"],
        "free_percent": pressure["free_percent"],
        "vm_stat": vm,
        "swap": swap,
    }
    raw["unified_memory_accounting"] = result
    return result


def _thermal_probe(raw: dict[str, Any]) -> dict[str, str]:
    pmset = shutil.which("pmset") or "/usr/bin/pmset"
    result = parse_thermal_pressure(
        _command("macos-thermal-pressure", [pmset, "-g", "therm"], raw)
    )
    raw["thermal_pressure"] = result
    return result


def bind_containment(
    platform_value: dict[str, Any],
    results: list[dict[str, Any]],
    mechanism: str | None,
) -> None:
    capabilities = platform_value["capabilities"]
    details = platform_value["details"]
    passed = mechanism in MACOS_CONTAINMENT_MECHANISMS
    capabilities["network_containment"] = "available" if passed else "unavailable"
    capabilities["process_containment"] = "available" if passed else "unavailable"
    detail = (
        f"{mechanism}; sandbox-inherited loopback-only networking, a private "
        "session/process group, bounded descendant settlement, and TERM/KILL cleanup"
        if passed
        else "runner did not provide an accepted macOS containment mechanism"
    )
    details["workload_containment"] = detail
    for result in results:
        if result.get("id") == "macos-workload-containment":
            result["status"] = "passed" if passed else "failed"
            result["details"] = detail
            break


def verify_contained_case(mechanism: str | None) -> dict[str, Any]:
    if mechanism not in MACOS_CONTAINMENT_MECHANISMS:
        raise PlatformProbeError("unrecognized macOS containment mechanism")
    listener = socket.socket()
    client = socket.socket()
    try:
        listener.bind(("127.0.0.1", 0))
        listener.listen()
        client.settimeout(1.0)
        client.connect(listener.getsockname())
        accepted, _address = listener.accept()
        accepted.close()
    finally:
        client.close()
        listener.close()
    external = socket.socket()
    try:
        external.setblocking(False)
        result = external.connect_ex(("192.0.2.1", 9))
        if result in {errno.EAGAIN, errno.EINPROGRESS, errno.EWOULDBLOCK}:
            _readable, writable, exceptional = select.select(
                [], [external], [external], 1.0
            )
            if not writable and not exceptional:
                raise PlatformProbeError(
                    "contained external route remained in progress instead of "
                    "returning a permission denial"
                )
            result = external.getsockopt(socket.SOL_SOCKET, socket.SO_ERROR)
    finally:
        external.close()
    if result not in {errno.EACCES, errno.EPERM}:
        raise PlatformProbeError(
            f"contained external route returned {result}, expected permission denial"
        )
    pid = os.getpid()
    process_group = os.getpgrp()
    session = os.getsid(0)
    if process_group != pid or session != pid:
        raise PlatformProbeError(
            f"contained process is not its session/group leader: "
            f"pid={pid}, pgrp={process_group}, sid={session}"
        )
    return {
        "mechanism": mechanism,
        "loopback_connect": "passed",
        "external_connect_errno": result,
        "pid": pid,
        "process_group": process_group,
        "session": session,
    }


def collect(
    selected_device: dict[str, Any],
    raw: dict[str, Any],
) -> tuple[dict[str, Any] | None, list[dict[str, Any]], list[str]]:
    if sys_platform() != "darwin":
        return None, [], []

    platform_raw: dict[str, Any] = {}
    raw["macos_platform"] = platform_raw
    capabilities = {key: "unavailable" for key in sorted(CAPABILITY_KEYS)}
    details: dict[str, str] = {}
    results: list[dict[str, Any]] = []
    unsupported: list[str] = []
    observations: dict[str, Any] = {
        "hardware_identity": None,
        "metal_runtime": None,
        "filesystem": None,
        "unified_memory": None,
        "memory_pressure": None,
        "thermal_pressure": None,
        "host_temperature": None,
        "gpu_temperature": None,
    }
    platform_value = {
        "kind": "macos",
        "capabilities": capabilities,
        "details": details,
        "observations": observations,
    }

    started = time.monotonic()
    try:
        hardware = _hardware_probe(selected_device, platform_raw)
        capabilities["apple_hardware_identity"] = "available"
        details["apple_hardware_identity"] = (
            f"{hardware['machine_name']} {hardware['machine_model']}; "
            f"{hardware['chip_type']}; {hardware['gpu_core_count']} GPU cores; "
            f"{hardware['physical_memory_bytes']} unified bytes"
        )
        observations["hardware_identity"] = hardware
        results.append(
            _result(
                "macos-apple-hardware-identity",
                required=True,
                passed=True,
                started=started,
                detail=details["apple_hardware_identity"],
            )
        )
    except (OSError, ValueError, PlatformProbeError) as exc:
        results.append(
            _result(
                "macos-apple-hardware-identity",
                required=True,
                passed=False,
                started=started,
                detail=str(exc),
            )
        )

    started = time.monotonic()
    try:
        runtime = _metal_runtime_probe(selected_device, platform_raw)
        capabilities["metal_runtime_identity"] = "available"
        details["metal_runtime_identity"] = (
            f"{runtime['name']}; unified={runtime['has_unified_memory']}; "
            f"recommended working set={runtime['recommended_max_working_set_bytes']}"
        )
        observations["metal_runtime"] = runtime
        results.append(
            _result(
                "macos-metal-runtime-identity",
                required=True,
                passed=True,
                started=started,
                detail=details["metal_runtime_identity"],
            )
        )
    except (OSError, PlatformProbeError) as exc:
        results.append(
            _result(
                "macos-metal-runtime-identity",
                required=True,
                passed=False,
                started=started,
                detail=str(exc),
            )
        )

    started = time.monotonic()
    try:
        toolchain = _toolchain_probe(platform_raw)
        capabilities["toolchain_provenance"] = "available"
        details["toolchain_provenance"] = (
            f"{toolchain['xcode_version'].replace(chr(10), '; ')}; "
            f"SDK {toolchain['sdk_version']} ({toolchain['sdk_build']})"
        )
        results.append(
            _result(
                "macos-toolchain-provenance",
                required=True,
                passed=True,
                started=started,
                detail=details["toolchain_provenance"],
            )
        )
    except (OSError, PlatformProbeError) as exc:
        results.append(
            _result(
                "macos-toolchain-provenance",
                required=True,
                passed=False,
                started=started,
                detail=str(exc),
            )
        )

    started = time.monotonic()
    try:
        compiler = _metal_compiler_probe(platform_raw)
        capabilities["metal_compiler"] = "available"
        details["metal_compiler"] = (
            f"compiled {compiler['air_bytes']}-byte AIR and linked "
            f"{compiler['metallib_bytes']}-byte metallib"
        )
        results.append(
            _result(
                "macos-metal-compiler",
                required=True,
                passed=True,
                started=started,
                detail=details["metal_compiler"],
            )
        )
    except (OSError, PlatformProbeError) as exc:
        results.append(
            _result(
                "macos-metal-compiler",
                required=True,
                passed=False,
                started=started,
                detail=str(exc),
            )
        )

    started = time.monotonic()
    try:
        filesystem = _filesystem_probe(platform_raw)
        capabilities["filesystem_semantics"] = "available"
        details["filesystem_semantics"] = (
            f"{filesystem['source']} {filesystem['fstype']} at "
            f"{filesystem['mount_point']}; atomic+durable+link probes passed; "
            f"case_sensitive={filesystem['case_sensitive']}"
        )
        observations["filesystem"] = filesystem
        results.append(
            _result(
                "macos-filesystem-semantics",
                required=True,
                passed=True,
                started=started,
                detail=details["filesystem_semantics"],
            )
        )
    except (OSError, PlatformProbeError) as exc:
        results.append(
            _result(
                "macos-filesystem-semantics",
                required=True,
                passed=False,
                started=started,
                detail=str(exc),
            )
        )

    started = time.monotonic()
    try:
        memory = _memory_probe(selected_device, platform_raw)
        capabilities["unified_memory_accounting"] = "available"
        capabilities["memory_pressure"] = "available"
        details["unified_memory_accounting"] = (
            f"total={memory['total_bytes']}; swap_total={memory['swap']['total_bytes']}; "
            f"compressed_pages={memory['vm_stat']['pages_occupied_by_compressor']}"
        )
        details["memory_pressure"] = (
            f"memory_pressure free={memory['free_percent']}%; "
            f"pageouts={memory['vm_stat']['pageouts']}; "
            f"swapouts={memory['vm_stat']['swapouts']}"
        )
        observations["unified_memory"] = {
            "total_bytes": memory["total_bytes"],
            "swap_total_bytes": memory["swap"]["total_bytes"],
            "swap_used_bytes": memory["swap"]["used_bytes"],
            "swap_encrypted": memory["swap"]["encrypted"],
        }
        observations["memory_pressure"] = {
            "free_percent": memory["free_percent"],
            "page_size_bytes": memory["vm_stat"]["page_size_bytes"],
            "pages_free": memory["vm_stat"]["pages_free"],
            "pages_active": memory["vm_stat"]["pages_active"],
            "pages_inactive": memory["vm_stat"]["pages_inactive"],
            "pages_wired_down": memory["vm_stat"]["pages_wired_down"],
            "pages_occupied_by_compressor": memory["vm_stat"][
                "pages_occupied_by_compressor"
            ],
            "pageins": memory["vm_stat"]["pageins"],
            "pageouts": memory["vm_stat"]["pageouts"],
            "swapins": memory["vm_stat"]["swapins"],
            "swapouts": memory["vm_stat"]["swapouts"],
        }
        results.append(
            _result(
                "macos-unified-memory-accounting",
                required=True,
                passed=True,
                started=started,
                detail=details["unified_memory_accounting"],
            )
        )
        results.append(
            _result(
                "macos-memory-pressure",
                required=True,
                passed=True,
                started=started,
                detail=details["memory_pressure"],
            )
        )
    except (OSError, PlatformProbeError) as exc:
        results.append(
            _result(
                "macos-unified-memory-accounting",
                required=True,
                passed=False,
                started=started,
                detail=str(exc),
            )
        )
        results.append(
            _result(
                "macos-memory-pressure",
                required=True,
                passed=False,
                started=started,
                detail=str(exc),
            )
        )

    started = time.monotonic()
    try:
        thermal = _thermal_probe(platform_raw)
        capabilities["thermal_pressure"] = "available"
        details["thermal_pressure"] = "; ".join(
            f"{key}={value}" for key, value in sorted(thermal.items())
        )
        observations["thermal_pressure"] = thermal
        results.append(
            _result(
                "macos-thermal-pressure",
                required=True,
                passed=True,
                started=started,
                detail=details["thermal_pressure"],
            )
        )
    except (OSError, PlatformProbeError) as exc:
        results.append(
            _result(
                "macos-thermal-pressure",
                required=True,
                passed=False,
                started=started,
                detail=str(exc),
            )
        )

    temperature_detail = (
        "macOS exposes thermal-pressure state here but no supported unprivileged "
        "host or GPU temperature API"
    )
    details["host_temperature"] = temperature_detail
    details["gpu_temperature"] = temperature_detail
    unsupported.extend(
        (
            "macos_host_temperature: " + temperature_detail,
            "macos_gpu_temperature: " + temperature_detail,
        )
    )
    for probe_id in ("macos-host-temperature", "macos-gpu-temperature"):
        results.append(
            _result(
                probe_id,
                required=False,
                passed=False,
                started=time.monotonic(),
                detail=temperature_detail,
            )
        )

    containment_detail = "awaiting runner containment binding"
    details["workload_containment"] = containment_detail
    results.append(
        _result(
            "macos-workload-containment",
            required=True,
            passed=False,
            started=time.monotonic(),
            detail=containment_detail,
        )
    )
    contained = os.environ.get(NETWORK_ISOLATION_ENV)
    if contained is not None:
        started = time.monotonic()
        try:
            platform_raw["contained_case"] = verify_contained_case(contained)
        except PlatformProbeError as exc:
            platform_raw["contained_case"] = {"error": str(exc)}
            bind_containment(platform_value, results, None)
            results.append(
                _result(
                    "macos-contained-case",
                    required=True,
                    passed=False,
                    started=started,
                    detail=str(exc),
                )
            )
        else:
            bind_containment(platform_value, results, contained)
            results.append(
                _result(
                    "macos-contained-case",
                    required=True,
                    passed=True,
                    started=started,
                    detail=(
                        "sandbox denied external networking, preserved loopback, "
                        "and the case owns its session/process group"
                    ),
                )
            )
    return platform_value, results, unsupported


def sys_platform() -> str:
    return sys.platform
