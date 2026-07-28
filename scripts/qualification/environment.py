#!/usr/bin/env python3
"""Capture a validated local accelerator environment receipt."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import platform
import re
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from receipt import atomic_write_json, validate_receipt
from source_tree_hash import HASH_FORMAT, SourceTreeHashError, source_tree_hash
import macos_platform
import wsl_platform


ROOT = Path(__file__).resolve().parents[2]
ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")
SENSITIVE_ENV_PARTS = (
    "AUTH",
    "TOKEN",
    "KEY",
    "SECRET",
    "PASSWORD",
    "CREDENTIAL",
    "WEBHOOK",
    "COOKIE",
)
CAPTURE_ENV_PREFIXES = (
    "KILN_",
    "HIP_",
    "HSA_",
    "ROCR_",
    "VK_",
    "GGML_VK_",
    "CUDA_",
    "NVIDIA_",
    "CUDARC_",
    "METAL_",
    "MTL_",
)


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def utc_text(value: datetime) -> str:
    return value.isoformat(timespec="milliseconds").replace("+00:00", "Z")


def sha256_bytes(value: bytes) -> str:
    return f"sha256:{hashlib.sha256(value).hexdigest()}"


def is_sensitive_environment_name(name: str) -> bool:
    normalized = name.upper()
    return any(part in normalized for part in SENSITIVE_ENV_PARTS)


def read_text(path: Path, default: str = "") -> str:
    try:
        return path.read_text(errors="replace").strip()
    except OSError:
        return default


def executable(name: str, *fallbacks: Path) -> str | None:
    found = shutil.which(name)
    if found:
        return found
    for fallback in fallbacks:
        if fallback.is_file() and os.access(fallback, os.X_OK):
            return str(fallback)
    return None


def run_probe(
    probe_id: str,
    argv: list[str],
    raw: dict[str, Any],
    *,
    required: bool = True,
    timeout: float = 30.0,
) -> tuple[dict[str, Any], str]:
    started = time.monotonic()
    if not argv or not argv[0]:
        duration = time.monotonic() - started
        raw[probe_id] = {"argv": argv, "returncode": None, "stdout": "", "stderr": "executable not found"}
        return (
            {
                "id": probe_id,
                "required": required,
                "status": "failed" if required else "skipped",
                "duration_seconds": duration,
                "metrics": [],
                "details": "executable not found",
            },
            "",
        )
    try:
        completed = subprocess.run(
            argv,
            cwd=ROOT,
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=timeout,
        )
        stdout = ANSI_RE.sub("", completed.stdout)
        stderr = ANSI_RE.sub("", completed.stderr)
        status = "passed" if completed.returncode == 0 else "failed"
        details = None if status == "passed" else f"exit {completed.returncode}: {stderr[-500:].strip()}"
        returncode: int | None = completed.returncode
    except subprocess.TimeoutExpired as exc:
        stdout = ANSI_RE.sub("", exc.stdout or "")
        stderr = ANSI_RE.sub("", exc.stderr or "")
        status = "failed"
        details = f"timed out after {timeout:g} seconds"
        returncode = None
    except OSError as exc:
        stdout = ""
        stderr = str(exc)
        status = "failed"
        details = str(exc)
        returncode = None
    duration = time.monotonic() - started
    raw[probe_id] = {
        "argv": argv,
        "returncode": returncode,
        "stdout": stdout,
        "stderr": stderr,
    }
    return (
        {
            "id": probe_id,
            "required": required,
            "status": status if required or status == "passed" else "skipped",
            "duration_seconds": duration,
            "metrics": [],
            "details": details,
        },
        stdout,
    )


def parse_os_release(path: Path = Path("/etc/os-release")) -> dict[str, str]:
    values: dict[str, str] = {}
    for line in read_text(path).splitlines():
        if "=" not in line or line.lstrip().startswith("#"):
            continue
        key, value = line.split("=", 1)
        values[key] = value.strip().strip('"')
    if not values and sys.platform == "darwin":
        return {
            "name": "macOS",
            "version": platform.mac_ver()[0] or "unknown",
            "kernel": platform.release() or "unknown",
            "architecture": platform.machine() or "unknown",
        }
    return {
        "name": values.get("NAME", platform.system() or "unknown"),
        "version": values.get("VERSION_ID") or values.get("BUILD_ID") or values.get("VERSION") or "unknown",
        "kernel": platform.release() or "unknown",
        "architecture": platform.machine() or "unknown",
    }


def captured_environment() -> dict[str, dict[str, Any]]:
    captured: dict[str, dict[str, Any]] = {}
    for key, value in sorted(os.environ.items()):
        if not key.startswith(CAPTURE_ENV_PREFIXES):
            continue
        sensitive = is_sensitive_environment_name(key)
        captured[key] = {
            "value": sha256_bytes(value.encode("utf-8")) if sensitive else value,
            "redacted": sensitive,
        }
    return captured


def git_commit() -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=ROOT,
        check=True,
        stdout=subprocess.PIPE,
        text=True,
    ).stdout.strip()


def git_clean() -> bool:
    return not subprocess.run(
        ["git", "status", "--porcelain", "--untracked-files=all"],
        cwd=ROOT,
        check=True,
        stdout=subprocess.PIPE,
        text=True,
    ).stdout.strip()


def _sysfs_number(path: Path) -> int | None:
    value = read_text(path)
    try:
        return int(value, 0)
    except ValueError:
        return None


def find_drm_device(vendor_id: int | None, device_id: int | None, root: Path = Path("/sys/class/drm")) -> Path | None:
    candidates: list[tuple[int, Path]] = []
    for card in sorted(root.glob("card[0-9]*")):
        device = card / "device"
        vendor = _sysfs_number(device / "vendor")
        product = _sysfs_number(device / "device")
        if vendor_id is not None and vendor != vendor_id:
            continue
        if device_id is not None and product != device_id:
            continue
        candidates.append((_sysfs_number(device / "mem_info_vram_total") or 0, device))
    return max(candidates, default=(0, None), key=lambda item: item[0])[1]


def drm_snapshot(device: Path | None) -> dict[str, Any]:
    if device is None:
        return {}
    keys = (
        "vendor",
        "device",
        "mem_info_vram_total",
        "mem_info_vram_used",
        "mem_info_gtt_total",
        "mem_info_gtt_used",
        "gpu_busy_percent",
        "power_dpm_force_performance_level",
    )
    return {key: read_text(device / key) for key in keys}


def parse_rocm_agent(text: str) -> dict[str, Any] | None:
    for block in re.split(r"\*{7}", text):
        if not re.search(r"Device Type:\s+GPU\b", block):
            continue
        def field(name: str) -> str:
            match = re.search(rf"^\s*{re.escape(name)}:\s*(.+?)\s*$", block, re.MULTILINE)
            return match.group(1).strip() if match else ""

        chip = field("Chip ID").split("(", 1)[0].strip()
        try:
            device_id = int(chip)
        except ValueError:
            device_id = None
        return {
            "architecture": field("Name"),
            "name": field("Marketing Name"),
            "device_id": device_id,
            "unified_memory": field("Memory Properties") == "APU",
            "compute_units": field("Compute Unit"),
            "wavefront_size": field("Wavefront Size").split("(", 1)[0].strip(),
        }
    return None


def parse_vulkan_summary(text: str) -> dict[str, Any] | None:
    block_match = re.search(
        r"^GPU(\d+):\s*$([\s\S]*?)(?=^GPU\d+:\s*$|\Z)",
        text,
        re.MULTILINE,
    )
    if not block_match:
        return None
    logical_index = int(block_match.group(1))
    block = block_match.group(2)

    def field(name: str) -> str:
        match = re.search(rf"^\s*{re.escape(name)}\s*=\s*(.+?)\s*$", block, re.MULTILINE)
        return match.group(1).strip() if match else ""

    def hex_field(name: str) -> int | None:
        try:
            return int(field(name), 0)
        except ValueError:
            return None

    name = field("deviceName")
    arch_match = re.search(r"\((?:RADV\s+)?([^()]+)\)\s*$", name)
    return {
        "logical_index": logical_index,
        "name": name,
        "architecture": arch_match.group(1).strip().lower().replace(" ", "_") if arch_match else f"pci-{field('vendorID')}-{field('deviceID')}",
        "vendor_id": hex_field("vendorID"),
        "device_id": hex_field("deviceID"),
        "integrated": field("deviceType") == "PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU",
        "api_version": field("apiVersion"),
        "driver_version": field("driverVersion"),
        "driver_id": field("driverID"),
        "driver_name": field("driverName"),
        "driver_info": field("driverInfo"),
    }


def parse_nvidia_smi_devices(text: str) -> list[dict[str, Any]]:
    devices: list[dict[str, Any]] = []
    for row_number, row in enumerate(csv.reader(text.splitlines()), start=1):
        fields = [field.strip() for field in row]
        if not fields or all(not field for field in fields):
            continue
        if len(fields) != 8:
            raise ValueError(
                f"nvidia-smi row {row_number} has {len(fields)} fields; expected 8"
            )
        index_text, name, uuid, pci_bus_id, capability, total_text, free_text, driver = fields
        try:
            logical_index = int(index_text)
            memory_mib = int(total_text)
            memory_free_mib = int(free_text)
        except ValueError as exc:
            raise ValueError(f"nvidia-smi row {row_number} has a non-integer field") from exc
        if logical_index < 0 or memory_mib <= 0 or not 0 <= memory_free_mib <= memory_mib:
            raise ValueError(f"nvidia-smi row {row_number} has invalid index or memory")
        match = re.fullmatch(r"(\d+)\.(\d+)", capability)
        if match is None:
            raise ValueError(
                f"nvidia-smi row {row_number} has invalid compute capability {capability!r}"
            )
        if not all((name, uuid, pci_bus_id, driver)):
            raise ValueError(f"nvidia-smi row {row_number} has an empty identity field")
        devices.append(
            {
                "logical_index": logical_index,
                "name": name,
                "device_uuid": uuid,
                "pci_bus_id": pci_bus_id,
                "compute_capability": capability,
                "architecture": f"sm_{match.group(1)}{match.group(2)}",
                "memory_bytes": memory_mib * 1024**2,
                "memory_available_bytes": memory_free_mib * 1024**2,
                "driver": driver,
            }
        )
    if len({device["logical_index"] for device in devices}) != len(devices):
        raise ValueError("nvidia-smi reported duplicate logical device indices")
    return devices


def parse_sw_vers(text: str) -> dict[str, str]:
    values: dict[str, str] = {}
    for line in text.splitlines():
        if ":" not in line:
            continue
        key, value = line.split(":", 1)
        values[key.strip()] = value.strip()
    return {
        "product_name": values.get("ProductName", "macOS"),
        "product_version": values.get("ProductVersion", "unknown"),
        "build_version": values.get("BuildVersion", "unknown"),
    }


def parse_metal_device(text: str, memory_text: str) -> dict[str, Any] | None:
    try:
        document = json.loads(text)
    except json.JSONDecodeError:
        return None
    displays = document.get("SPDisplaysDataType") if isinstance(document, dict) else None
    if not isinstance(displays, list):
        return None
    for logical_index, display in enumerate(displays):
        if not isinstance(display, dict):
            continue
        name = display.get("sppci_model") or display.get("_name")
        if not isinstance(name, str) or not name.strip():
            continue
        metal_support = next(
            (
                value
                for key, value in display.items()
                if ("metal" in key.lower() or "mtl" in key.lower())
                and isinstance(value, str)
                and value.strip()
            ),
            None,
        )
        if metal_support is None:
            continue
        try:
            memory_bytes = int(memory_text.strip())
        except ValueError:
            memory_bytes = 0
        if memory_bytes <= 0:
            return None
        core_text = display.get("sppci_cores") or display.get("spdisplays_gpu_cores")
        core_match = re.search(r"\d+", str(core_text)) if core_text is not None else None
        normalized_name = re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")
        return {
            "logical_index": logical_index,
            "name": name.strip(),
            "architecture": normalized_name or "apple_gpu",
            "memory_bytes": memory_bytes,
            "memory_available_bytes": None,
            "compute_units": int(core_match.group()) if core_match else None,
            "metal_support": metal_support.strip(),
        }
    return None


def parse_nvcc_version(text: str) -> str:
    match = re.search(r"release\s+([^,\s]+),\s+V(\S+)", text)
    return f"release {match.group(1)}, V{match.group(2)}" if match else first_line(text)


def first_line(text: str) -> str:
    return next((line.strip() for line in text.splitlines() if line.strip()), "unknown")


def collect_backend(
    backend: str,
    raw: dict[str, Any],
    *,
    device_index: int | None = None,
) -> tuple[dict[str, Any], dict[str, str], dict[str, str], list[dict[str, Any]]]:
    results: list[dict[str, Any]] = []
    home = Path.home()
    rustc = executable("rustc", home / ".cargo/bin/rustc")
    cargo = executable("cargo", home / ".cargo/bin/cargo")
    rust_result, rust_text = run_probe("rustc-version", [rustc or "", "--version", "--verbose"], raw)
    cargo_result, cargo_text = run_probe("cargo-version", [cargo or "", "--version"], raw)
    results.extend((rust_result, cargo_result))
    compiler = {"rustc": first_line(rust_text), "cargo": first_line(cargo_text)}

    if backend == "rocm":
        rocminfo = executable("rocminfo", Path("/opt/rocm/bin/rocminfo"))
        hipcc = executable("hipcc", Path("/opt/rocm/bin/hipcc"))
        rocm_result, rocm_text = run_probe("rocm-device-probe", [rocminfo or ""], raw)
        hipcc_result, hipcc_text = run_probe("hipcc-version", [hipcc or "", "--version"], raw)
        results.extend((rocm_result, hipcc_result))
        agent = parse_rocm_agent(rocm_text)
        if agent is None:
            results.append(
                {
                    "id": "rocm-gpu-agent",
                    "required": True,
                    "status": "failed",
                    "duration_seconds": 0.0,
                    "metrics": [],
                    "details": "rocminfo did not report a GPU agent",
                }
            )
            agent = {"architecture": "unavailable", "name": "unavailable", "device_id": None, "unified_memory": False}
        else:
            results.append(
                {
                    "id": "rocm-gpu-agent",
                    "required": True,
                    "status": "passed",
                    "duration_seconds": 0.0,
                    "metrics": [],
                    "details": f"{agent['architecture']} {agent['name']}",
                }
            )
        drm = find_drm_device(0x1002, agent.get("device_id"))
        raw["drm"] = drm_snapshot(drm)
        memory = _sysfs_number(drm / "mem_info_vram_total") if drm else None
        memory_used = _sysfs_number(drm / "mem_info_vram_used") if drm else None
        memory_available = (
            memory - memory_used
            if memory is not None and memory_used is not None and 0 <= memory_used <= memory
            else None
        )
        rocm_version = read_text(Path("/opt/rocm/.info/version"), "unknown")
        runtime_match = re.search(r"^Runtime Version:\s*(\S+)", rocm_text, re.MULTILINE)
        runtime = {
            "rocm": rocm_version,
            "hsa": runtime_match.group(1) if runtime_match else "unknown",
            "hipcc": first_line(hipcc_text),
        }
        compiler["hipcc"] = first_line(hipcc_text)
        device = {
            "name": agent["name"],
            "architecture": agent["architecture"],
            "memory_bytes": memory,
            "memory_available_bytes": memory_available,
            "unified_memory": bool(agent["unified_memory"]),
            "driver": f"amdgpu kernel {platform.release()}",
            "logical_index": None,
            "device_uuid": None,
            "pci_bus_id": None,
            "compute_capability": None,
            "compute_units": (
                int(agent["compute_units"])
                if str(agent.get("compute_units", "")).isdigit()
                else None
            ),
        }
    elif backend == "vulkan":
        vulkaninfo = executable("vulkaninfo")
        glslc = executable("glslc") or executable("glslangValidator")
        vk_result, vk_text = run_probe("vulkan-device-probe", [vulkaninfo or "", "--summary"], raw)
        shader_result, shader_text = run_probe("shader-compiler-version", [glslc or "", "--version"], raw)
        results.extend((vk_result, shader_result))
        parsed = parse_vulkan_summary(vk_text)
        if parsed is None:
            results.append(
                {
                    "id": "vulkan-physical-device",
                    "required": True,
                    "status": "failed",
                    "duration_seconds": 0.0,
                    "metrics": [],
                    "details": "vulkaninfo did not report a physical device",
                }
            )
            parsed = {
                "name": "unavailable",
                "architecture": "unavailable",
                "vendor_id": None,
                "device_id": None,
                "integrated": False,
                "api_version": "unknown",
                "driver_version": "unknown",
                "driver_id": "unknown",
                "driver_name": "unknown",
                "driver_info": "unknown",
            }
        else:
            results.append(
                {
                    "id": "vulkan-physical-device",
                    "required": True,
                    "status": "passed",
                    "duration_seconds": 0.0,
                    "metrics": [],
                    "details": parsed["name"],
                }
            )
        drm = find_drm_device(parsed.get("vendor_id"), parsed.get("device_id"))
        raw["drm"] = drm_snapshot(drm)
        memory = _sysfs_number(drm / "mem_info_vram_total") if drm else None
        memory_used = _sysfs_number(drm / "mem_info_vram_used") if drm else None
        memory_available = (
            memory - memory_used
            if memory is not None and memory_used is not None and 0 <= memory_used <= memory
            else None
        )
        instance_match = re.search(r"Vulkan Instance Version:\s*(\S+)", vk_text)
        runtime = {
            "vulkan_instance": instance_match.group(1) if instance_match else "unknown",
            "vulkan_device": parsed["api_version"] or "unknown",
            "driver": parsed["driver_info"] or parsed["driver_version"] or "unknown",
        }
        compiler["shader"] = first_line(shader_text)
        device = {
            "name": parsed["name"],
            "architecture": parsed["architecture"],
            "memory_bytes": memory,
            "memory_available_bytes": memory_available,
            "unified_memory": bool(parsed["integrated"]),
            "driver": f"{parsed['driver_name']} {parsed['driver_info']}".strip(),
            "logical_index": parsed.get("logical_index"),
            "device_uuid": None,
            "pci_bus_id": None,
            "compute_capability": None,
            "compute_units": None,
        }
    elif backend == "cuda":
        nvidia_smi = executable("nvidia-smi", Path("/usr/bin/nvidia-smi"))
        cuda_roots = [
            Path(value)
            for name in ("CUDA_ROOT", "CUDA_HOME", "CUDA_PATH")
            if (value := os.environ.get(name))
        ]
        nvcc = executable(
            "nvcc",
            *(root / "bin/nvcc" for root in cuda_roots),
            Path("/usr/local/cuda/bin/nvcc"),
        )
        query = (
            "index,name,uuid,pci.bus_id,compute_cap,memory.total,"
            "memory.free,driver_version"
        )
        smi_result, smi_text = run_probe(
            "cuda-device-probe",
            [
                nvidia_smi or "",
                f"--query-gpu={query}",
                "--format=csv,noheader,nounits",
            ],
            raw,
        )
        nvcc_result, nvcc_text = run_probe(
            "nvcc-version", [nvcc or "", "--version"], raw
        )
        results.extend((smi_result, nvcc_result))
        selected_index = 0 if device_index is None else device_index
        try:
            cuda_devices = parse_nvidia_smi_devices(smi_text)
        except ValueError as exc:
            cuda_devices = []
            parse_error = str(exc)
        else:
            parse_error = None
        raw["cuda_devices"] = cuda_devices
        selected = next(
            (
                candidate
                for candidate in cuda_devices
                if candidate["logical_index"] == selected_index
            ),
            None,
        )
        if selected is None:
            available = ", ".join(str(item["logical_index"]) for item in cuda_devices) or "none"
            detail = parse_error or (
                f"nvidia-smi did not report requested logical index {selected_index}; "
                f"available indices: {available}"
            )
            results.append(
                {
                    "id": "cuda-selected-device",
                    "required": True,
                    "status": "failed",
                    "duration_seconds": 0.0,
                    "metrics": [],
                    "details": detail,
                }
            )
            selected = {
                "logical_index": selected_index,
                "name": "unavailable",
                "architecture": "unavailable",
                "memory_bytes": None,
                "memory_available_bytes": None,
                "device_uuid": None,
                "pci_bus_id": None,
                "compute_capability": None,
                "driver": "unavailable",
            }
        else:
            results.append(
                {
                    "id": "cuda-selected-device",
                    "required": True,
                    "status": "passed",
                    "duration_seconds": 0.0,
                    "metrics": [],
                    "details": (
                        f"logical index {selected_index}: {selected['name']} "
                        f"({selected['architecture']}, {selected['memory_bytes']} bytes)"
                    ),
                }
            )
        toolkit = parse_nvcc_version(nvcc_text)
        compiler["nvcc"] = toolkit
        runtime = {
            "cuda_driver": selected["driver"],
            "cuda_toolkit": toolkit,
            "nvidia_smi_query": query,
        }
        device = {
            "name": selected["name"],
            "architecture": selected["architecture"],
            "memory_bytes": selected["memory_bytes"],
            "memory_available_bytes": selected["memory_available_bytes"],
            "unified_memory": False,
            "driver": f"NVIDIA {selected['driver']}",
            "logical_index": selected["logical_index"],
            "device_uuid": selected["device_uuid"],
            "pci_bus_id": selected["pci_bus_id"],
            "compute_capability": selected["compute_capability"],
            "compute_units": None,
        }
    elif backend == "metal":
        system_profiler = executable(
            "system_profiler", Path("/usr/sbin/system_profiler")
        )
        sysctl = executable("sysctl", Path("/usr/sbin/sysctl"))
        sw_vers = executable("sw_vers", Path("/usr/bin/sw_vers"))
        xcrun = executable("xcrun", Path("/usr/bin/xcrun"))
        displays_result, displays_text = run_probe(
            "metal-device-probe",
            [system_profiler or "", "SPDisplaysDataType", "-json"],
            raw,
            timeout=90.0,
        )
        memory_result, memory_text = run_probe(
            "unified-memory-probe", [sysctl or "", "-n", "hw.memsize"], raw
        )
        os_result, sw_vers_text = run_probe(
            "macos-version", [sw_vers or ""], raw
        )
        metal_result, metal_text = run_probe(
            "metal-compiler-path", [xcrun or "", "--find", "metal"], raw
        )
        sdk_result, sdk_text = run_probe(
            "macos-sdk-version",
            [xcrun or "", "--sdk", "macosx", "--show-sdk-version"],
            raw,
        )
        clang_result, clang_text = run_probe(
            "apple-clang-version", [xcrun or "", "clang", "--version"], raw
        )
        results.extend(
            (
                displays_result,
                memory_result,
                os_result,
                metal_result,
                sdk_result,
                clang_result,
            )
        )
        parsed = parse_metal_device(displays_text, memory_text)
        if parsed is None:
            results.append(
                {
                    "id": "metal-selected-device",
                    "required": True,
                    "status": "failed",
                    "duration_seconds": 0.0,
                    "metrics": [],
                    "details": (
                        "system_profiler did not report a Metal-capable GPU "
                        "and positive unified-memory total"
                    ),
                }
            )
            parsed = {
                "logical_index": 0,
                "name": "unavailable",
                "architecture": "unavailable",
                "memory_bytes": None,
                "memory_available_bytes": None,
                "compute_units": None,
                "metal_support": "unavailable",
            }
        else:
            results.append(
                {
                    "id": "metal-selected-device",
                    "required": True,
                    "status": "passed",
                    "duration_seconds": 0.0,
                    "metrics": [],
                    "details": (
                        f"logical index {parsed['logical_index']}: {parsed['name']} "
                        f"({parsed['memory_bytes']} unified bytes)"
                    ),
                }
            )
        macos = parse_sw_vers(sw_vers_text)
        compiler["metal"] = first_line(metal_text)
        compiler["apple_clang"] = first_line(clang_text)
        runtime = {
            "metal": parsed["metal_support"],
            "macos": macos["product_version"],
            "macos_build": macos["build_version"],
            "macos_sdk": first_line(sdk_text),
        }
        device = {
            "name": parsed["name"],
            "architecture": parsed["architecture"],
            "memory_bytes": parsed["memory_bytes"],
            "memory_available_bytes": parsed["memory_available_bytes"],
            "unified_memory": True,
            "driver": (
                f"{macos['product_name']} {macos['product_version']} "
                f"({macos['build_version']})"
            ),
            "logical_index": parsed["logical_index"],
            "device_uuid": None,
            "pci_bus_id": None,
            "compute_capability": None,
            "compute_units": parsed["compute_units"],
        }
    else:
        raise ValueError(f"unsupported accelerator backend: {backend}")
    return device, runtime, compiler, results


def device_expectation_result(
    device: dict[str, Any],
    *,
    expected_name_regex: str | None,
    expected_compute_units: int | None,
    minimum_memory_mib: int | None,
    maximum_memory_mib: int | None,
) -> dict[str, Any] | None:
    if (
        expected_name_regex is None
        and expected_compute_units is None
        and minimum_memory_mib is None
        and maximum_memory_mib is None
    ):
        return None
    failures: list[str] = []
    name = device.get("name")
    if expected_name_regex is not None and (
        not isinstance(name, str) or re.fullmatch(expected_name_regex, name) is None
    ):
        failures.append(
            f"device name {name!r} does not fully match {expected_name_regex!r}"
        )
    compute_units = device.get("compute_units")
    if expected_compute_units is not None and compute_units != expected_compute_units:
        failures.append(
            f"device compute-unit count {compute_units!r} does not equal {expected_compute_units}"
        )
    memory = device.get("memory_bytes")
    if not isinstance(memory, int) or isinstance(memory, bool) or memory <= 0:
        if minimum_memory_mib is not None or maximum_memory_mib is not None:
            failures.append("device total memory is unavailable")
    else:
        memory_mib = memory // 1024**2
        if minimum_memory_mib is not None and memory_mib < minimum_memory_mib:
            failures.append(
                f"device total memory {memory_mib} MiB is below {minimum_memory_mib} MiB"
            )
        if maximum_memory_mib is not None and memory_mib > maximum_memory_mib:
            failures.append(
                f"device total memory {memory_mib} MiB exceeds {maximum_memory_mib} MiB"
            )
    return {
        "id": "device-class-expectation",
        "required": True,
        "status": "failed" if failures else "passed",
        "duration_seconds": 0.0,
        "metrics": [],
        "details": (
            "; ".join(failures)
            if failures
            else "selected device matches the committed class"
        ),
    }


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--backend", choices=("rocm", "vulkan", "cuda", "metal"), required=True
    )
    parser.add_argument("--host-id", required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--device-index", type=int)
    parser.add_argument("--expected-device-name-regex")
    parser.add_argument("--expected-compute-units", type=int)
    parser.add_argument("--minimum-memory-mib", type=int)
    parser.add_argument("--maximum-memory-mib", type=int)
    args = parser.parse_args(argv)
    if args.device_index is not None and args.device_index < 0:
        parser.error("--device-index must be non-negative")
    if args.expected_compute_units is not None and args.expected_compute_units <= 0:
        parser.error("--expected-compute-units must be positive")
    for name in ("minimum_memory_mib", "maximum_memory_mib"):
        value = getattr(args, name)
        if value is not None and value <= 0:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    if (
        args.minimum_memory_mib is not None
        and args.maximum_memory_mib is not None
        and args.minimum_memory_mib > args.maximum_memory_mib
    ):
        parser.error("--minimum-memory-mib cannot exceed --maximum-memory-mib")
    if args.expected_device_name_regex is not None:
        try:
            re.compile(args.expected_device_name_regex)
        except re.error as exc:
            parser.error(f"--expected-device-name-regex is invalid: {exc}")
    return args


def main(argv: list[str] | None = None) -> int:
    raw_argv = sys.argv[1:] if argv is None else argv
    args = parse_args(raw_argv)
    started_at = utc_now()
    started_monotonic = time.monotonic()
    clean_at_start = git_clean()
    commit = git_commit()
    try:
        tree_hash, _ = source_tree_hash(ROOT)
    except SourceTreeHashError as exc:
        print(f"environment qualification failed: {exc}", file=sys.stderr)
        return 1

    timestamp = started_at.strftime("%Y%m%dT%H%M%SZ").lower()
    receipt_id = f"{timestamp}-{args.backend}-{args.host_id}-environment-v1"
    output = args.output or Path(
        f"qualification/receipts/{args.backend}/{args.host_id}/{receipt_id}.json"
    )
    output = output if output.is_absolute() else ROOT / output
    if output.exists():
        print(f"refusing to overwrite existing receipt: {output}", file=sys.stderr)
        return 1

    raw: dict[str, Any] = {
        "schema_version": 1,
        "receipt_id": receipt_id,
        "captured_environment": captured_environment(),
        "device_selection": {
            "logical_index": args.device_index,
            "expected_name_regex": args.expected_device_name_regex,
            "expected_compute_units": args.expected_compute_units,
            "minimum_memory_mib": args.minimum_memory_mib,
            "maximum_memory_mib": args.maximum_memory_mib,
        },
    }
    device, runtime, compiler, results = collect_backend(
        args.backend, raw, device_index=args.device_index
    )
    platform_value: dict[str, Any] | None = None
    unsupported: list[str] = []
    if args.backend == "cuda":
        platform_value, platform_results, unsupported = wsl_platform.collect(
            device,
            raw,
        )
        results.extend(platform_results)
    elif args.backend == "metal":
        platform_value, platform_results, unsupported = macos_platform.collect(
            device,
            raw,
        )
        results.extend(platform_results)
    expectation = device_expectation_result(
        device,
        expected_name_regex=args.expected_device_name_regex,
        expected_compute_units=args.expected_compute_units,
        minimum_memory_mib=args.minimum_memory_mib,
        maximum_memory_mib=args.maximum_memory_mib,
    )
    if expectation is not None:
        results.append(expectation)
    results.insert(
        0,
        {
            "id": "clean-source-at-start",
            "required": True,
            "status": "passed" if clean_at_start else "failed",
            "duration_seconds": 0.0,
            "metrics": [],
            "details": None if clean_at_start else "Git worktree was dirty before capture",
        },
    )

    raw_path = ROOT / ".qualification" / "runs" / receipt_id / "environment-probes.json"
    atomic_write_json(raw_path, raw)
    raw_bytes = raw_path.read_bytes()
    finished_at = utc_now()
    duration = time.monotonic() - started_monotonic
    passed = all(not result["required"] or result["status"] == "passed" for result in results)
    command = [sys.executable, "scripts/qualification/environment.py", *raw_argv]
    receipt = {
        "schema_version": 1,
        "receipt_id": receipt_id,
        "created_at_utc": utc_text(finished_at),
        "source": {
            "tree_hash_format": HASH_FORMAT,
            "tree_hash": tree_hash,
            "git_commit": commit,
            "git_worktree_clean": clean_at_start,
        },
        "qualification": {
            "kind": "environment",
            "backend": args.backend,
            "profile": "environment-v1",
            "verdict": "passed" if passed else "failed",
            "started_at_utc": utc_text(started_at),
            "finished_at_utc": utc_text(finished_at),
            "duration_seconds": duration,
            "command": command,
        },
        "environment": {
            "host_id": args.host_id,
            "os": parse_os_release(),
            "device": device,
            "runtime": runtime,
            "compiler": compiler,
            **({"platform": platform_value} if platform_value is not None else {}),
        },
        "model": None,
        "workload": None,
        "effective_config": {
            "device_selection": {
                "logical_index": args.device_index,
                "expected_name_regex": args.expected_device_name_regex,
                "expected_compute_units": args.expected_compute_units,
                "minimum_memory_mib": args.minimum_memory_mib,
                "maximum_memory_mib": args.maximum_memory_mib,
            }
        },
        "results": results,
        "metrics": [],
        "artifacts": [
            {
                "kind": "environment_probes",
                "location": "local_ignored",
                "path": str(raw_path.relative_to(ROOT)),
                "sha256": sha256_bytes(raw_bytes),
                "bytes": len(raw_bytes),
            }
        ],
        "unsupported": unsupported,
        "notes": [],
    }
    errors = validate_receipt(receipt, root=ROOT, require_local_artifacts=True)
    if errors:
        print("generated receipt failed validation:", file=sys.stderr)
        for error in errors:
            print(f"  - {error}", file=sys.stderr)
        return 1
    atomic_write_json(output, receipt)
    print(output.relative_to(ROOT) if output.is_relative_to(ROOT) else output)
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
