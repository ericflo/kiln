#!/usr/bin/env python3
"""Capture a validated local ROCm or Vulkan environment receipt."""

from __future__ import annotations

import argparse
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


ROOT = Path(__file__).resolve().parents[2]
ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")
SENSITIVE_ENV_PARTS = ("TOKEN", "KEY", "SECRET", "PASSWORD", "CREDENTIAL", "WEBHOOK_URL")
CAPTURE_ENV_PREFIXES = ("KILN_", "HIP_", "HSA_", "ROCR_", "VK_", "GGML_VK_")


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def utc_text(value: datetime) -> str:
    return value.isoformat(timespec="milliseconds").replace("+00:00", "Z")


def sha256_bytes(value: bytes) -> str:
    return f"sha256:{hashlib.sha256(value).hexdigest()}"


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
        sensitive = any(part in key.upper() for part in SENSITIVE_ENV_PARTS)
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
    block_match = re.search(r"^GPU\d+:\s*$([\s\S]*?)(?=^GPU\d+:\s*$|\Z)", text, re.MULTILINE)
    if not block_match:
        return None
    block = block_match.group(1)

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


def first_line(text: str) -> str:
    return next((line.strip() for line in text.splitlines() if line.strip()), "unknown")


def collect_backend(backend: str, raw: dict[str, Any]) -> tuple[dict[str, Any], dict[str, str], dict[str, str], list[dict[str, Any]]]:
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
            "unified_memory": bool(agent["unified_memory"]),
            "driver": f"amdgpu kernel {platform.release()}",
        }
    else:
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
            "unified_memory": bool(parsed["integrated"]),
            "driver": f"{parsed['driver_name']} {parsed['driver_info']}".strip(),
        }
    return device, runtime, compiler, results


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=("rocm", "vulkan"), required=True)
    parser.add_argument("--host-id", required=True)
    parser.add_argument("--output", type=Path)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(sys.argv[1:] if argv is None else argv)
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
    output = args.output or Path(f"qualification/receipts/{args.backend}/{args.host_id}/{receipt_id}.json")
    output = output if output.is_absolute() else ROOT / output
    if output.exists():
        print(f"refusing to overwrite existing receipt: {output}", file=sys.stderr)
        return 1

    raw: dict[str, Any] = {
        "schema_version": 1,
        "receipt_id": receipt_id,
        "captured_environment": captured_environment(),
    }
    device, runtime, compiler, results = collect_backend(args.backend, raw)
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
    command = [sys.executable, "scripts/qualification/environment.py", "--backend", args.backend, "--host-id", args.host_id]
    if args.output is not None:
        command.extend(("--output", str(args.output)))
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
        },
        "model": None,
        "workload": None,
        "effective_config": {"environment": captured_environment()},
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
        "unsupported": [],
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
