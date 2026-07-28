#!/usr/bin/env python3
"""Collect fail-closed WSL2 platform and CUDA bridge provenance."""

from __future__ import annotations

import ctypes
import errno
import hashlib
import json
import os
import platform
import re
import shutil
import socket
import stat
import subprocess
import tempfile
import time
import uuid
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
NETWORK_ISOLATION_ENV = "KILN_QUALIFICATION_NETWORK_ISOLATION"
WSL_INTEROP = Path("/proc/sys/fs/binfmt_misc/WSLInterop")
WSL_LIB = Path("/usr/lib/wsl/lib")
CUDA_LINK = Path("/usr/local/cuda")
POWERSHELL = Path(
    "/mnt/c/Windows/System32/WindowsPowerShell/v1.0/powershell.exe"
)
WSL_EXE = Path("/mnt/c/Windows/System32/wsl.exe")
CAPABILITY_KEYS = {
    "wsl_identity",
    "windows_identity",
    "driver_identity",
    "cuda_driver_bridge",
    "cuda_toolkit",
    "cuda_runtime",
    "nvml",
    "filesystem_semantics",
    "network_containment",
    "process_containment",
    "systemd_system",
    "systemd_user_transient",
    "cgroup_memory_delegation",
    "memory_accounting",
    "host_temperature",
    "gpu_temperature",
}
CAPABILITY_STATUSES = {"available", "unavailable"}
WSL_CONTAINMENT_MECHANISMS = {
    "util-linux-unshare-user-net-pid-landlock-v1",
}


class PlatformProbeError(RuntimeError):
    """A WSL platform probe did not establish its claimed capability."""


class NvmlMemory(ctypes.Structure):
    _fields_ = [
        ("total", ctypes.c_ulonglong),
        ("free", ctypes.c_ulonglong),
        ("used", ctypes.c_ulonglong),
    ]


def is_wsl2(
    *,
    kernel_release: str | None = None,
    interop_path: Path = WSL_INTEROP,
) -> bool:
    release = platform.release() if kernel_release is None else kernel_release
    try:
        interop = interop_path.read_text(errors="replace")
    except OSError:
        interop = ""
    contained = os.environ.get(NETWORK_ISOLATION_ENV)
    return "microsoft-standard-wsl2" in release.lower() and (
        "enabled" in interop or contained in WSL_CONTAINMENT_MECHANISMS
    )


def decode_windows_output(value: bytes) -> str:
    if value.startswith((b"\xff\xfe", b"\xfe\xff")):
        return value.decode("utf-16").lstrip("\ufeff")
    if b"\x00" in value[:256]:
        return value.decode("utf-16-le").lstrip("\ufeff")
    return value.decode("utf-8-sig")


def parse_windows_thermal_zones(text: str) -> list[dict[str, Any]]:
    try:
        raw = json.loads(text)
    except json.JSONDecodeError as exc:
        raise PlatformProbeError(
            f"Windows formatted thermal telemetry is malformed JSON: {exc}"
        ) from exc
    rows = raw if isinstance(raw, list) else [raw]
    if not rows:
        raise PlatformProbeError("Windows formatted thermal telemetry is empty")
    normalized: list[dict[str, Any]] = []
    names: set[str] = set()
    keys = {
        "Name",
        "Temperature",
        "HighPrecisionTemperature",
        "PercentPassiveLimit",
        "ThrottleReasons",
    }
    for index, row in enumerate(rows):
        if not isinstance(row, dict) or set(row) != keys:
            raise PlatformProbeError(
                f"Windows formatted thermal row {index} has invalid keys"
            )
        name = row["Name"]
        integer_fields = {key: row[key] for key in keys - {"Name"}}
        if (
            not isinstance(name, str)
            or not name
            or name != name.strip()
            or name in names
            or any(
                not isinstance(value, int) or isinstance(value, bool)
                for value in integer_fields.values()
            )
        ):
            raise PlatformProbeError(
                f"Windows formatted thermal row {index} has invalid values"
            )
        kelvin = integer_fields["Temperature"]
        tenths_kelvin = integer_fields["HighPrecisionTemperature"]
        percent_passive_limit = integer_fields["PercentPassiveLimit"]
        throttle_reasons = integer_fields["ThrottleReasons"]
        if not 1 <= kelvin <= 1000 or not 1 <= tenths_kelvin <= 10_000:
            raise PlatformProbeError(
                f"Windows formatted thermal row {index} is implausible"
            )
        if not (
            0 <= percent_passive_limit <= 0xFFFFFFFF
            and 0 <= throttle_reasons <= 0xFFFFFFFF
        ):
            raise PlatformProbeError(
                f"Windows formatted thermal row {index} has invalid counters"
            )
        if abs(kelvin * 10 - tenths_kelvin) > 10:
            raise PlatformProbeError(
                f"Windows formatted thermal row {index} precision fields disagree"
            )
        millicelsius = tenths_kelvin * 100 - 273_150
        if not -50_000 <= millicelsius <= 200_000:
            raise PlatformProbeError(
                f"Windows formatted thermal row {index} converted implausibly"
            )
        names.add(name)
        normalized.append(
            {
                "name": name,
                "temperature_kelvin": kelvin,
                "high_precision_temperature_tenths_kelvin": tenths_kelvin,
                "temperature_millicelsius": millicelsius,
                "percent_passive_limit": percent_passive_limit,
                "throttle_reasons": throttle_reasons,
            }
        )
    return normalized


def parse_wsl_version(text: str) -> dict[str, str]:
    result: dict[str, str] = {}
    names = {
        "wsl version": "wsl_version",
        "kernel version": "kernel_version",
        "wslg version": "wslg_version",
        "msrdc version": "msrdc_version",
        "direct3d version": "direct3d_version",
        "dxcore version": "dxcore_version",
        "windows version": "windows_version",
    }
    for line in text.splitlines():
        if ":" not in line:
            continue
        label, value = line.split(":", 1)
        key = names.get(label.strip().lower())
        if key is not None and value.strip():
            result[key] = value.strip()
    required = {"wsl_version", "kernel_version", "windows_version"}
    if not required <= result.keys():
        raise PlatformProbeError(
            "wsl.exe --version omitted " + ", ".join(sorted(required - result.keys()))
        )
    return result


def windows_nvidia_driver_version(value: str) -> str:
    parts = value.split(".")
    if len(parts) != 4 or any(not part.isdigit() for part in parts):
        raise PlatformProbeError(f"invalid Windows display driver version {value!r}")
    encoded = (int(parts[2]) % 10) * 10_000 + int(parts[3])
    if encoded <= 0:
        raise PlatformProbeError(f"invalid Windows display driver version {value!r}")
    return f"{encoded // 100}.{encoded % 100:02d}"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return f"sha256:{digest.hexdigest()}"


def _file_identity(path: Path, *, within: Path | None = None) -> dict[str, Any]:
    try:
        resolved = path.resolve(strict=True)
    except OSError as exc:
        raise PlatformProbeError(f"cannot resolve {path}: {exc}") from exc
    if within is not None:
        try:
            resolved.relative_to(within.resolve(strict=True))
        except (OSError, ValueError) as exc:
            raise PlatformProbeError(f"{path} resolves outside {within}") from exc
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
) -> str:
    try:
        completed = subprocess.run(
            argv,
            cwd=ROOT,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
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
    try:
        stdout = decode_windows_output(completed.stdout)
        stderr = decode_windows_output(completed.stderr)
    except UnicodeDecodeError as exc:
        raise PlatformProbeError(f"{probe_id} returned undecodable output") from exc
    raw[probe_id] = {
        "argv": argv,
        "returncode": completed.returncode,
        "stdout": stdout,
        "stderr": stderr,
    }
    if completed.returncode != 0:
        detail = stderr.strip()[-500:]
        raise PlatformProbeError(
            f"{probe_id} exited {completed.returncode}"
            + (f": {detail}" if detail else "")
        )
    return stdout


def _call_zero(function: Any, *args: Any) -> None:
    code = int(function(*args))
    if code != 0:
        raise PlatformProbeError(f"{function.__name__} returned {code}")


def _cuda_bridge_probe(raw: dict[str, Any]) -> dict[str, Any]:
    files = {
        name: _file_identity(WSL_LIB / name, within=WSL_LIB)
        for name in ("nvidia-smi", "libcuda.so.1", "libnvidia-ml.so.1")
    }
    cuda = ctypes.CDLL(str(WSL_LIB / "libcuda.so.1"))
    _call_zero(cuda.cuInit, 0)
    driver_api = ctypes.c_int()
    _call_zero(cuda.cuDriverGetVersion, ctypes.byref(driver_api))
    result = {"files": files, "driver_api_version": driver_api.value}
    raw["cuda_driver_bridge"] = result
    return result


def _cuda_toolkit_probe(raw: dict[str, Any]) -> dict[str, Any]:
    root = CUDA_LINK.resolve(strict=True)
    if not root.is_dir() or root.parent != Path("/usr/local"):
        raise PlatformProbeError(f"{CUDA_LINK} has unexpected target {root}")
    version_path = root / "version.json"
    identity = _file_identity(version_path, within=root)
    try:
        document = json.loads(version_path.read_bytes())
        sdk = document["cuda"]["version"]
        cudart_manifest = document["cuda_cudart"]["version"]
        nvcc_manifest = document["cuda_nvcc"]["version"]
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, KeyError, TypeError) as exc:
        raise PlatformProbeError(f"invalid CUDA version manifest: {exc}") from exc
    if not all(
        isinstance(value, str) and re.fullmatch(r"\d+\.\d+(?:\.\d+)?", value)
        for value in (sdk, cudart_manifest, nvcc_manifest)
    ):
        raise PlatformProbeError("CUDA version manifest has invalid versions")
    nvcc_path = root / "bin/nvcc"
    nvcc_identity = _file_identity(nvcc_path, within=root)
    nvcc_text = _command(
        "wsl2-nvcc-version",
        [str(nvcc_path), "--version"],
        raw,
    )
    match = re.search(r"release\s+(\d+\.\d+),\s+V(\d+\.\d+\.\d+)", nvcc_text)
    if match is None:
        raise PlatformProbeError("nvcc did not report a strict release and build")
    if match.group(1) != ".".join(nvcc_manifest.split(".")[:2]):
        raise PlatformProbeError("nvcc release disagrees with version.json")
    if match.group(2) != nvcc_manifest:
        raise PlatformProbeError("nvcc build disagrees with version.json")
    result = {
        "root": str(root),
        "manifest": identity,
        "sdk_version": sdk,
        "cudart_manifest_version": cudart_manifest,
        "nvcc_manifest_version": nvcc_manifest,
        "nvcc": nvcc_identity,
    }
    raw["cuda_toolkit"] = result
    return result


def _format_cuda_api(value: int) -> str:
    if value <= 0:
        raise PlatformProbeError(f"invalid CUDA API version {value}")
    return f"{value // 1000}.{(value % 1000) // 10}"


def _cuda_runtime_probe(toolkit: dict[str, Any], raw: dict[str, Any]) -> dict[str, Any]:
    root = Path(toolkit["root"])
    candidates = sorted(
        (root / "targets/x86_64-linux/lib").glob("libcudart.so.*.*.*")
    )
    if len(candidates) != 1:
        raise PlatformProbeError(
            f"expected one concrete libcudart, found {len(candidates)}"
        )
    cudart_path = candidates[0]
    identity = _file_identity(cudart_path, within=root)
    cudart = ctypes.CDLL(str(cudart_path))
    runtime_api = ctypes.c_int()
    driver_api = ctypes.c_int()
    _call_zero(cudart.cudaRuntimeGetVersion, ctypes.byref(runtime_api))
    _call_zero(cudart.cudaDriverGetVersion, ctypes.byref(driver_api))
    runtime_version = _format_cuda_api(runtime_api.value)
    if runtime_version != ".".join(toolkit["cudart_manifest_version"].split(".")[:2]):
        raise PlatformProbeError("libcudart API disagrees with version.json")
    result = {
        "library": identity,
        "runtime_api": runtime_api.value,
        "runtime_version": runtime_version,
        "driver_api": driver_api.value,
        "driver_api_version": _format_cuda_api(driver_api.value),
    }
    raw["cuda_runtime"] = result
    return result


def _nvml_text(function: Any, *args: Any, size: int = 128) -> str:
    buffer = ctypes.create_string_buffer(size)
    _call_zero(function, *args, buffer, size)
    return buffer.value.decode("utf-8")


def _nvml_probe(device_index: int, raw: dict[str, Any]) -> dict[str, Any]:
    nvml = ctypes.CDLL(str(WSL_LIB / "libnvidia-ml.so.1"))
    init = getattr(nvml, "nvmlInit_v2", nvml.nvmlInit)
    shutdown = nvml.nvmlShutdown
    _call_zero(init)
    try:
        nvml_version = _nvml_text(nvml.nvmlSystemGetNVMLVersion)
        driver_version = _nvml_text(nvml.nvmlSystemGetDriverVersion)
        count = ctypes.c_uint()
        get_count = getattr(nvml, "nvmlDeviceGetCount_v2", nvml.nvmlDeviceGetCount)
        _call_zero(get_count, ctypes.byref(count))
        if device_index < 0 or device_index >= count.value:
            raise PlatformProbeError(
                f"NVML device index {device_index} is outside count {count.value}"
            )
        handle = ctypes.c_void_p()
        get_handle = getattr(
            nvml,
            "nvmlDeviceGetHandleByIndex_v2",
            nvml.nvmlDeviceGetHandleByIndex,
        )
        _call_zero(get_handle, device_index, ctypes.byref(handle))
        device_uuid = _nvml_text(nvml.nvmlDeviceGetUUID, handle)
        name = _nvml_text(nvml.nvmlDeviceGetName, handle)
        memory = NvmlMemory()
        _call_zero(nvml.nvmlDeviceGetMemoryInfo, handle, ctypes.byref(memory))
        temperature = ctypes.c_uint()
        _call_zero(
            nvml.nvmlDeviceGetTemperature,
            handle,
            0,
            ctypes.byref(temperature),
        )
    finally:
        _call_zero(shutdown)
    result = {
        "version": nvml_version,
        "driver_version": driver_version,
        "device_count": count.value,
        "logical_index": device_index,
        "device_uuid": device_uuid,
        "name": name,
        "memory_total_bytes": memory.total,
        "memory_free_bytes": memory.free,
        "memory_used_bytes": memory.used,
        "temperature_c": temperature.value,
    }
    raw["nvml"] = result
    return result


def _filesystem_probe(raw: dict[str, Any]) -> dict[str, Any]:
    findmnt = shutil.which("findmnt") or "/usr/bin/findmnt"
    text = _command(
        "wsl2-filesystem-mount",
        [findmnt, "--json", "--target", str(ROOT)],
        raw,
    )
    try:
        filesystems = json.loads(text)["filesystems"]
        mount = filesystems[0]
        source = mount["source"]
        fstype = mount["fstype"]
        target = mount["target"]
    except (json.JSONDecodeError, KeyError, IndexError, TypeError) as exc:
        raise PlatformProbeError(f"invalid findmnt JSON: {exc}") from exc
    if fstype != "ext4" or not str(source).startswith("/dev/"):
        raise PlatformProbeError(
            f"qualification root must be native ext4, got {source} {fstype}"
        )
    if ROOT == Path("/mnt") or Path("/mnt") in ROOT.parents:
        raise PlatformProbeError(f"qualification root is Windows-mounted: {ROOT}")
    probe_root = ROOT / ".qualification" / "wsl-platform-probes"
    probe_root.mkdir(parents=True, exist_ok=True)
    directory = Path(tempfile.mkdtemp(prefix="filesystem-", dir=probe_root))
    try:
        original = directory / "Case"
        different_case = directory / "case"
        replacement = directory / "replacement"
        original.write_bytes(b"original")
        different_case.write_bytes(b"different")
        with original.open("rb") as handle:
            os.fsync(handle.fileno())
        os.link(original, directory / "hardlink")
        original_inode = os.stat(original).st_ino
        if original_inode != os.stat(directory / "hardlink").st_ino:
            raise PlatformProbeError("hardlink did not preserve inode identity")
        os.symlink("Case", directory / "symlink")
        replacement.write_bytes(b"replacement")
        with replacement.open("rb") as handle:
            os.fsync(handle.fileno())
        os.replace(replacement, original)
        directory_fd = os.open(
            directory,
            os.O_RDONLY | getattr(os, "O_DIRECTORY", 0),
        )
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
        if original.read_bytes() != b"replacement":
            raise PlatformProbeError("atomic replacement did not publish new bytes")
        if different_case.read_bytes() != b"different":
            raise PlatformProbeError("filesystem is not case-sensitive")
        if (directory / "symlink").resolve() != original.resolve():
            raise PlatformProbeError("relative symlink did not resolve")
        if (directory / "hardlink").read_bytes() != b"original":
            raise PlatformProbeError("atomic replacement mutated the linked old inode")
        if os.stat(original).st_ino == original_inode:
            raise PlatformProbeError("atomic replacement retained the old inode")
    finally:
        shutil.rmtree(directory, ignore_errors=True)
    result = {
        "root": str(ROOT),
        "source": str(source),
        "fstype": str(fstype),
        "mount_target": str(target),
        "atomic_replace": True,
        "directory_fsync": True,
        "hardlink": True,
        "symlink": True,
        "case_sensitive": True,
    }
    raw["filesystem_semantics"] = result
    return result


def _meminfo() -> dict[str, int]:
    result: dict[str, int] = {}
    for line in Path("/proc/meminfo").read_text().splitlines():
        match = re.fullmatch(r"([A-Za-z_()]+):\s+(\d+)\s+kB", line)
        if match:
            result[match.group(1)] = int(match.group(2)) * 1024
    required = {"MemTotal", "MemAvailable", "SwapTotal", "SwapFree"}
    if not required <= result.keys():
        raise PlatformProbeError(
            "/proc/meminfo omitted " + ", ".join(sorted(required - result.keys()))
        )
    if (
        result["MemTotal"] <= 0
        or not 0 <= result["MemAvailable"] <= result["MemTotal"]
        or result["SwapTotal"] < 0
        or not 0 <= result["SwapFree"] <= result["SwapTotal"]
    ):
        raise PlatformProbeError("/proc/meminfo reported inconsistent capacity")
    return {key: result[key] for key in sorted(required)}


def _cgroup_probe(raw: dict[str, Any]) -> dict[str, Any]:
    if Path("/sys/fs/cgroup/cgroup.controllers").read_text().strip() == "":
        raise PlatformProbeError("cgroup v2 root exposes no controllers")
    candidates = sorted(
        Path("/sys/fs/cgroup/user.slice").glob(
            "user-*.slice/user@*.service/app.slice"
        )
    )
    writable = [
        path
        for path in candidates
        if os.access(path, os.W_OK)
        and {"memory", "pids"}
        <= set((path / "cgroup.controllers").read_text().split())
    ]
    if len(writable) != 1:
        raise PlatformProbeError(
            f"expected one writable user app.slice with memory+pids, found {len(writable)}"
        )
    parent = writable[0]
    child = parent / f"kiln-probe-{os.getpid()}-{uuid.uuid4().hex}"
    child.mkdir()
    try:
        (child / "memory.max").write_text(str(64 * 1024**2))
        (child / "memory.swap.max").write_text("0")
        memory_max = (child / "memory.max").read_text().strip()
        swap_max = (child / "memory.swap.max").read_text().strip()
        if memory_max != str(64 * 1024**2) or swap_max != "0":
            raise PlatformProbeError("delegated cgroup limits did not round-trip")
    finally:
        child.rmdir()
    result = {
        "version": 2,
        "delegated_parent": str(parent),
        "controllers": sorted((parent / "cgroup.controllers").read_text().split()),
        "memory_current_bytes": int((parent / "memory.current").read_text()),
        "memory_max": (parent / "memory.max").read_text().strip(),
        "memory_swap_current_bytes": int(
            (parent / "memory.swap.current").read_text()
        ),
        "memory_swap_max": (parent / "memory.swap.max").read_text().strip(),
        "probe_memory_max_bytes": 64 * 1024**2,
        "probe_swap_max_bytes": 0,
    }
    raw["cgroup_memory_delegation"] = result
    return result


def _hwmon_temperatures(
    root: Path = Path("/sys/class/hwmon"),
) -> list[dict[str, Any]]:
    sensors: list[dict[str, Any]] = []
    for device in sorted(root.glob("hwmon*")):
        name_path = device / "name"
        if not name_path.is_file():
            continue
        try:
            name = name_path.read_text(errors="replace").strip()
        except OSError as exc:
            raise PlatformProbeError(
                f"cannot read hwmon name {name_path}: {exc}"
            ) from exc
        if not name:
            raise PlatformProbeError(f"empty hwmon name at {name_path}")
        for input_path in sorted(device.glob("temp*_input")):
            try:
                value = int(input_path.read_text().strip())
            except (OSError, ValueError) as exc:
                raise PlatformProbeError(
                    f"cannot read hwmon temperature {input_path}: {exc}"
                ) from exc
            if not -50_000 <= value <= 200_000:
                raise PlatformProbeError(
                    f"implausible hwmon temperature {value} at {input_path}"
                )
            label_path = input_path.with_name(
                input_path.name.replace("_input", "_label")
            )
            try:
                label = (
                    label_path.read_text(errors="replace").strip()
                    if label_path.is_file()
                    else input_path.stem
                )
                resolved_input = input_path.resolve(strict=True)
            except OSError as exc:
                raise PlatformProbeError(
                    f"cannot identify hwmon temperature {input_path}: {exc}"
                ) from exc
            if not label:
                raise PlatformProbeError(f"empty hwmon label at {label_path}")
            sensors.append(
                {
                    "hwmon_name": name,
                    "label": label,
                    "input_path": str(resolved_input),
                    "temperature_millicelsius": value,
                }
            )
    return sensors


def bind_containment(
    platform_value: dict[str, Any],
    results: list[dict[str, Any]],
    mechanism: str | None,
) -> None:
    capabilities = platform_value["capabilities"]
    details = platform_value["details"]
    passed = mechanism in WSL_CONTAINMENT_MECHANISMS
    capabilities["network_containment"] = "available" if passed else "unavailable"
    capabilities["process_containment"] = "available" if passed else "unavailable"
    detail = (
        f"{mechanism}; loopback-only network, private PID namespace, "
        "kill-child lifetime, and Landlock-denied WSL interop"
        if passed
        else "runner did not provide an accepted WSL2 containment mechanism"
    )
    details["workload_containment"] = detail
    for result in results:
        if result.get("id") == "wsl2-workload-containment":
            result["status"] = "passed" if passed else "failed"
            result["details"] = detail
            break


def verify_contained_case(mechanism: str | None) -> dict[str, Any]:
    if mechanism not in WSL_CONTAINMENT_MECHANISMS:
        raise PlatformProbeError("unrecognized WSL2 containment mechanism")
    interfaces = [name for _index, name in socket.if_nameindex()]
    if interfaces != ["lo"]:
        raise PlatformProbeError(
            f"contained interfaces are not exactly ['lo']: {interfaces!r}"
        )
    external = socket.socket()
    try:
        external.settimeout(0.5)
        result = external.connect_ex(("192.0.2.1", 9))
    finally:
        external.close()
    if result not in {errno.ENETUNREACH, errno.EHOSTUNREACH}:
        raise PlatformProbeError(
            f"contained external route returned {result}, expected unreachable"
        )
    uid_map = Path("/proc/self/uid_map").read_text().strip()
    if not re.search(r"^\s*0\s+[1-9][0-9]*\s+1\s*$", uid_map, re.MULTILINE):
        raise PlatformProbeError(
            "process is not root-mapped in a private user namespace: "
            f"{uid_map!r}"
        )
    try:
        subprocess.run(
            ["/mnt/c/Windows/System32/cmd.exe", "/d", "/c", "ver"],
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
            timeout=5,
        )
    except PermissionError:
        pass
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise PlatformProbeError(
            f"Windows interop rejection was not a permission denial: {exc}"
        ) from exc
    else:
        raise PlatformProbeError("Windows interop execution was not denied")
    return {
        "mechanism": mechanism,
        "interfaces": interfaces,
        "external_connect_errno": result,
        "uid_map": uid_map,
        "windows_interop_execute": "denied",
    }


def collect(
    selected_device: dict[str, Any],
    raw: dict[str, Any],
) -> tuple[dict[str, Any] | None, list[dict[str, Any]], list[str]]:
    if not is_wsl2():
        return None, [], []

    platform_raw: dict[str, Any] = {}
    raw["wsl2_platform"] = platform_raw
    capabilities = {key: "unavailable" for key in sorted(CAPABILITY_KEYS)}
    details: dict[str, str] = {}
    results: list[dict[str, Any]] = []
    unsupported: list[str] = []
    platform_value = {
        "kind": "wsl2",
        "capabilities": capabilities,
        "details": details,
        "observations": {
            "host_temperatures": [],
            "gpu_temperature": None,
        },
    }
    observations = platform_value["observations"]

    started = time.monotonic()
    kernel = platform.release()
    try:
        interop = WSL_INTEROP.read_text(errors="replace").strip()
    except OSError:
        interop = "hidden_in_private_proc"
    capabilities["wsl_identity"] = "available"
    details["wsl_identity"] = f"kernel={kernel}; interop={interop}"
    results.append(
        _result(
            "wsl2-identity",
            required=True,
            passed=True,
            started=started,
            detail=details["wsl_identity"],
        )
    )

    contained = os.environ.get(NETWORK_ISOLATION_ENV)
    contained_verified = False
    if contained:
        try:
            platform_raw["contained_case"] = verify_contained_case(contained)
            contained_verified = True
        except PlatformProbeError as exc:
            platform_raw["contained_case"] = {"error": str(exc)}
    windows: dict[str, Any] | None = None
    started = time.monotonic()
    if contained:
        detail = "Windows interop is intentionally unavailable inside the contained case"
        unsupported.append("wsl2_contained_case_windows_identity: " + detail)
        results.append(
            _result(
                "wsl2-windows-identity",
                required=False,
                passed=False,
                started=started,
                detail=detail,
            )
        )
    else:
        try:
            wsl_versions = parse_wsl_version(
                _command(
                    "wsl2-wsl-version",
                    [str(WSL_EXE), "--version"],
                    platform_raw,
                )
            )
            script = (
                "$ErrorActionPreference='Stop';"
                "$os=Get-CimInstance Win32_OperatingSystem|"
                "Select-Object Caption,Version,BuildNumber;"
                "$gpu=@(Get-CimInstance Win32_VideoController|"
                "Select-Object Name,DriverVersion,PNPDeviceID);"
                "[pscustomobject]@{os=$os;gpu=$gpu}|"
                "ConvertTo-Json -Compress -Depth 4"
            )
            windows = json.loads(
                _command(
                    "wsl2-windows-cim",
                    [
                        str(POWERSHELL),
                        "-NoLogo",
                        "-NoProfile",
                        "-NonInteractive",
                        "-Command",
                        script,
                    ],
                    platform_raw,
                )
            )
            if not isinstance(windows, dict) or not isinstance(windows.get("os"), dict):
                raise PlatformProbeError("Windows CIM output has invalid shape")
            os_value = windows["os"]
            for key in ("Caption", "Version", "BuildNumber"):
                if not isinstance(os_value.get(key), str) or not os_value[key]:
                    raise PlatformProbeError(
                        f"Windows CIM output omitted os.{key}"
                    )
            platform_raw["wsl_versions"] = wsl_versions
            platform_raw["windows_identity"] = windows
            capabilities["windows_identity"] = "available"
            details["windows_identity"] = (
                f"{os_value.get('Caption')} {os_value.get('Version')} "
                f"build {os_value.get('BuildNumber')}; WSL {wsl_versions['wsl_version']}"
            )
            results.append(
                _result(
                    "wsl2-windows-identity",
                    required=True,
                    passed=True,
                    started=started,
                    detail=details["windows_identity"],
                )
            )
        except (OSError, json.JSONDecodeError, PlatformProbeError) as exc:
            detail = str(exc)
            unsupported.append("wsl2_windows_identity: " + detail)
            results.append(
                _result(
                    "wsl2-windows-identity",
                    required=True,
                    passed=False,
                    started=started,
                    detail=detail,
                )
            )

    started = time.monotonic()
    try:
        bridge = _cuda_bridge_probe(platform_raw)
        capabilities["cuda_driver_bridge"] = "available"
        details["cuda_driver_bridge"] = (
            f"libcuda API {bridge['driver_api_version']}; "
            f"bridge_root={WSL_LIB}"
        )
        results.append(
            _result(
                "wsl2-cuda-driver-bridge",
                required=True,
                passed=True,
                started=started,
                detail=details["cuda_driver_bridge"],
            )
        )
    except (OSError, PlatformProbeError) as exc:
        bridge = None
        results.append(
            _result(
                "wsl2-cuda-driver-bridge",
                required=True,
                passed=False,
                started=started,
                detail=str(exc),
            )
        )

    started = time.monotonic()
    try:
        toolkit = _cuda_toolkit_probe(platform_raw)
        capabilities["cuda_toolkit"] = "available"
        details["cuda_toolkit"] = (
            f"SDK {toolkit['sdk_version']}; nvcc {toolkit['nvcc_manifest_version']}; "
            f"root={toolkit['root']}"
        )
        results.append(
            _result(
                "wsl2-cuda-toolkit",
                required=True,
                passed=True,
                started=started,
                detail=details["cuda_toolkit"],
            )
        )
    except (OSError, PlatformProbeError) as exc:
        toolkit = None
        results.append(
            _result(
                "wsl2-cuda-toolkit",
                required=True,
                passed=False,
                started=started,
                detail=str(exc),
            )
        )

    started = time.monotonic()
    try:
        if toolkit is None:
            raise PlatformProbeError("CUDA toolkit provenance failed")
        runtime = _cuda_runtime_probe(toolkit, platform_raw)
        if bridge is not None and runtime["driver_api"] != bridge["driver_api_version"]:
            raise PlatformProbeError("libcuda and libcudart driver APIs disagree")
        capabilities["cuda_runtime"] = "available"
        details["cuda_runtime"] = (
            f"runtime API {runtime['runtime_version']}; "
            f"driver API {runtime['driver_api_version']}"
        )
        results.append(
            _result(
                "wsl2-cuda-runtime",
                required=True,
                passed=True,
                started=started,
                detail=details["cuda_runtime"],
            )
        )
    except (OSError, PlatformProbeError) as exc:
        results.append(
            _result(
                "wsl2-cuda-runtime",
                required=True,
                passed=False,
                started=started,
                detail=str(exc),
            )
        )

    started = time.monotonic()
    try:
        logical_index = selected_device.get("logical_index")
        if not isinstance(logical_index, int) or isinstance(logical_index, bool):
            raise PlatformProbeError("selected CUDA logical index is unavailable")
        nvml = _nvml_probe(logical_index, platform_raw)
        for key in ("device_uuid", "name"):
            if nvml[key] != selected_device.get(key):
                raise PlatformProbeError(
                    f"NVML {key} {nvml[key]!r} disagrees with nvidia-smi "
                    f"{selected_device.get(key)!r}"
                )
        selected_driver = str(selected_device.get("driver", "")).removeprefix(
            "NVIDIA "
        )
        if nvml["driver_version"] != selected_driver:
            raise PlatformProbeError("NVML and nvidia-smi driver versions disagree")
        if nvml["memory_total_bytes"] != selected_device.get("memory_bytes"):
            raise PlatformProbeError("NVML and nvidia-smi total memory disagree")
        if (
            nvml["memory_free_bytes"] < 0
            or nvml["memory_used_bytes"] < 0
            or nvml["memory_free_bytes"] + nvml["memory_used_bytes"]
            != nvml["memory_total_bytes"]
        ):
            raise PlatformProbeError("NVML memory accounting is inconsistent")
        if not 1 <= nvml["temperature_c"] <= 150:
            raise PlatformProbeError(
                f"NVML temperature is implausible: {nvml['temperature_c']} C"
            )
        capabilities["nvml"] = "available"
        capabilities["gpu_temperature"] = "available"
        details["nvml"] = (
            f"NVML {nvml['version']}; driver {nvml['driver_version']}; "
            f"UUID {nvml['device_uuid']}"
        )
        details["gpu_temperature"] = (
            f"selected device {nvml['device_uuid']}: "
            f"{nvml['temperature_c']} C at capture"
        )
        observations["gpu_temperature"] = {
            "source": "nvml",
            "device_uuid": nvml["device_uuid"],
            "temperature_millicelsius": nvml["temperature_c"] * 1000,
        }
        results.append(
            _result(
                "wsl2-nvml-identity",
                required=True,
                passed=True,
                started=started,
                detail=details["nvml"],
            )
        )
        results.append(
            _result(
                "wsl2-gpu-temperature",
                required=True,
                passed=True,
                started=started,
                detail=details["gpu_temperature"],
            )
        )
    except (OSError, PlatformProbeError) as exc:
        detail = str(exc)
        details["gpu_temperature"] = detail
        unsupported.append("wsl2_gpu_temperature: " + detail)
        results.append(
            _result(
                "wsl2-nvml-identity",
                required=True,
                passed=False,
                started=started,
                detail=detail,
            )
        )
        results.append(
            _result(
                "wsl2-gpu-temperature",
                required=True,
                passed=False,
                started=started,
                detail=detail,
            )
        )

    started = time.monotonic()
    try:
        if windows is None:
            raise PlatformProbeError("Windows GPU identity is unavailable")
        gpu_value = windows.get("gpu")
        gpu_items = gpu_value if isinstance(gpu_value, list) else [gpu_value]
        matches = [
            item
            for item in gpu_items
            if isinstance(item, dict)
            and item.get("Name") == selected_device.get("name")
        ]
        if len(matches) != 1:
            raise PlatformProbeError(
                f"expected one matching Windows GPU, found {len(matches)}"
            )
        windows_driver = matches[0].get("DriverVersion")
        if not isinstance(windows_driver, str):
            raise PlatformProbeError("Windows GPU driver version is unavailable")
        normalized = windows_nvidia_driver_version(windows_driver)
        selected_driver = str(selected_device.get("driver", "")).removeprefix(
            "NVIDIA "
        )
        if normalized != selected_driver:
            raise PlatformProbeError(
                f"Windows driver {windows_driver} maps to {normalized}, "
                f"not nvidia-smi {selected_driver}"
            )
        capabilities["driver_identity"] = "available"
        details["driver_identity"] = (
            f"Windows {windows_driver}; NVIDIA {normalized}; "
            f"PNP {matches[0].get('PNPDeviceID')}"
        )
        results.append(
            _result(
                "wsl2-windows-driver-crosscheck",
                required=True,
                passed=True,
                started=started,
                detail=details["driver_identity"],
            )
        )
    except PlatformProbeError as exc:
        results.append(
            _result(
                "wsl2-windows-driver-crosscheck",
                required=not bool(contained),
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
            f"{filesystem['mount_target']}; atomic+durable+link+case probes passed"
        )
        results.append(
            _result(
                "wsl2-filesystem-semantics",
                required=True,
                passed=True,
                started=started,
                detail=details["filesystem_semantics"],
            )
        )
    except (OSError, PlatformProbeError) as exc:
        results.append(
            _result(
                "wsl2-filesystem-semantics",
                required=True,
                passed=False,
                started=started,
                detail=str(exc),
            )
        )

    started = time.monotonic()
    try:
        memory = _meminfo()
        platform_raw["memory_accounting"] = memory
        capabilities["memory_accounting"] = "available"
        details["memory_accounting"] = "; ".join(
            f"{key}={value}" for key, value in memory.items()
        )
        results.append(
            _result(
                "wsl2-memory-accounting",
                required=True,
                passed=True,
                started=started,
                detail=details["memory_accounting"],
            )
        )
    except (OSError, PlatformProbeError) as exc:
        results.append(
            _result(
                "wsl2-memory-accounting",
                required=True,
                passed=False,
                started=started,
                detail=str(exc),
            )
        )

    started = time.monotonic()
    try:
        cgroup = _cgroup_probe(platform_raw)
        capabilities["cgroup_memory_delegation"] = "available"
        details["cgroup_memory_delegation"] = (
            f"{cgroup['delegated_parent']}; controllers="
            f"{','.join(cgroup['controllers'])}; memory.max and swap.max round-trip"
        )
        results.append(
            _result(
                "wsl2-cgroup-memory-delegation",
                required=True,
                passed=True,
                started=started,
                detail=details["cgroup_memory_delegation"],
            )
        )
    except (OSError, ValueError, PlatformProbeError) as exc:
        results.append(
            _result(
                "wsl2-cgroup-memory-delegation",
                required=True,
                passed=False,
                started=started,
                detail=str(exc),
            )
        )

    started = time.monotonic()
    pid1 = Path("/proc/1/comm").read_text(errors="replace").strip()
    if pid1 == "systemd":
        try:
            state = _command(
                "wsl2-systemd-system",
                ["/usr/bin/systemctl", "is-system-running"],
                platform_raw,
            ).strip()
            if state != "running":
                raise PlatformProbeError(f"system manager state is {state!r}")
            capabilities["systemd_system"] = "available"
            details["systemd_system"] = "PID 1 systemd; system state running"
            systemd_passed = True
        except PlatformProbeError as exc:
            details["systemd_system"] = str(exc)
            systemd_passed = False
    else:
        details["systemd_system"] = (
            f"PID 1 is {pid1!r} inside the qualification PID namespace"
        )
        systemd_passed = False
    results.append(
        _result(
            "wsl2-systemd-system",
            required=not bool(contained),
            passed=systemd_passed,
            started=started,
            detail=details["systemd_system"],
        )
    )
    if not systemd_passed:
        unsupported.append("wsl2_systemd_system: " + details["systemd_system"])

    started = time.monotonic()
    if contained:
        details["systemd_user_transient"] = (
            "user-manager launch is intentionally forbidden inside the "
            "contained case because it would execute outside Landlock and the "
            "private namespaces"
        )
        user_passed = False
    else:
        try:
            state = _command(
                "wsl2-systemd-user-state",
                ["/usr/bin/systemctl", "--user", "is-system-running"],
                platform_raw,
            ).strip()
            if state != "running":
                raise PlatformProbeError(f"user manager state is {state!r}")
            user_bus = Path(f"/run/user/{os.getuid()}/bus")
            metadata = user_bus.stat()
            if not stat.S_ISSOCK(metadata.st_mode) or metadata.st_uid != os.getuid():
                raise PlatformProbeError(
                    f"user bus is not an owned socket: {user_bus}"
                )
            _command(
                "wsl2-systemd-user-transient",
                [
                    "/usr/bin/env",
                    f"DBUS_SESSION_BUS_ADDRESS=unix:path={user_bus}",
                    "/usr/bin/systemd-run",
                    "--user",
                    "--wait",
                    "--pipe",
                    "--collect",
                    "--",
                    "/usr/bin/true",
                ],
                platform_raw,
            )
            capabilities["systemd_user_transient"] = "available"
            details["systemd_user_transient"] = (
                "user manager running; owned user bus and transient unit passed"
            )
            user_passed = True
        except (OSError, PlatformProbeError) as exc:
            details["systemd_user_transient"] = str(exc)
            user_passed = False
    results.append(
        _result(
            "wsl2-systemd-user-transient",
            required=False,
            passed=user_passed,
            started=started,
            detail=details["systemd_user_transient"],
        )
    )
    if not user_passed:
        unsupported.append(
            "wsl2_systemd_user_transient: "
            + details["systemd_user_transient"]
        )

    started = time.monotonic()
    linux_error: str | None = None
    try:
        sensors = _hwmon_temperatures()
    except PlatformProbeError as exc:
        sensors = []
        linux_error = str(exc)
        linux_detail = str(exc)
    else:
        linux_detail = "no readable Linux hwmon temperature inputs"
    platform_raw["host_temperature_sensors"] = sensors
    if sensors:
        capabilities["host_temperature"] = "available"
        observations["host_temperatures"] = [
            {
                "source": "linux_hwmon",
                "name": (
                    f"{sensor['hwmon_name']}/{sensor['label']} "
                    f"[{sensor['input_path']}]"
                ),
                "temperature_millicelsius": sensor[
                    "temperature_millicelsius"
                ],
            }
            for sensor in sensors
        ]
        details["host_temperature"] = (
            f"{len(sensors)} readable Linux hwmon temperature inputs"
        )
        host_temperature_passed = True
    elif linux_error is not None:
        details["host_temperature"] = linux_error
        host_temperature_passed = False
    elif not contained:
        script = (
            "$ErrorActionPreference='Stop';"
            "Get-CimInstance -ClassName "
            "Win32_PerfFormattedData_Counters_ThermalZoneInformation|"
            "Select-Object Name,Temperature,HighPrecisionTemperature,"
            "PercentPassiveLimit,ThrottleReasons|"
            "ConvertTo-Json -Compress -Depth 3"
        )
        try:
            thermal_text = _command(
                "wsl2-windows-formatted-temperature",
                [
                    str(POWERSHELL),
                    "-NoLogo",
                    "-NoProfile",
                    "-NonInteractive",
                    "-Command",
                    script,
                ],
                platform_raw,
            ).strip()
            windows_sensors = parse_windows_thermal_zones(thermal_text)
            platform_raw["host_temperature_sensors"] = windows_sensors
            capabilities["host_temperature"] = "available"
            observations["host_temperatures"] = [
                {
                    "source": "windows_formatted_thermal_zone",
                    "name": sensor["name"],
                    "temperature_millicelsius": sensor[
                        "temperature_millicelsius"
                    ],
                }
                for sensor in windows_sensors
            ]
            details["host_temperature"] = (
                "Windows formatted thermal provider: "
                + ", ".join(
                    f"{sensor['name']}="
                    f"{sensor['temperature_millicelsius']}mC"
                    for sensor in windows_sensors
                )
            )
            host_temperature_passed = True
        except PlatformProbeError as exc:
            details["host_temperature"] = (
                f"{linux_detail}; Windows formatted thermal provider "
                f"unavailable: {exc}"
            )
            host_temperature_passed = False
    else:
        details["host_temperature"] = (
            f"{linux_detail}; Windows interop is intentionally unavailable "
            "inside the contained case"
        )
        host_temperature_passed = False
    results.append(
        _result(
            "wsl2-host-temperature",
            required=not bool(contained),
            passed=host_temperature_passed,
            started=started,
            detail=details["host_temperature"],
        )
    )
    if not host_temperature_passed:
        unsupported.append(
            "wsl2_host_temperature: " + details["host_temperature"]
        )

    started = time.monotonic()
    containment_result = _result(
        "wsl2-workload-containment",
        required=True,
        passed=False,
        started=started,
        detail="runner did not provide an accepted WSL2 containment mechanism",
    )
    results.append(containment_result)
    bind_containment(
        platform_value,
        results,
        contained if contained_verified else None,
    )

    if set(capabilities) != CAPABILITY_KEYS:
        raise AssertionError("WSL capability collector emitted an inconsistent key set")
    return platform_value, results, unsupported
