#!/usr/bin/env python3
"""Validate compact local hardware qualification receipts."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from source_tree_hash import HASH_FORMAT, SourceTreeHashError, source_tree_hash
from strict_json import (
    StrictJSONError,
    loads as strict_json_loads,
)


ROOT = Path(__file__).resolve().parents[2]
SCHEMA_VERSION = 1
SHA256_RE = re.compile(r"^sha256:[0-9a-f]{64}$")
COMMIT_RE = re.compile(r"^[0-9a-f]{40}$")
ID_RE = re.compile(r"^[a-z0-9][a-z0-9._-]{2,127}$")
RESULT_ID_RE = re.compile(r"^[a-z0-9][a-z0-9._-]{0,127}$")
METRIC_NAME_RE = re.compile(r"^[a-z][a-z0-9_.-]{0,127}$")
HOST_ID_RE = re.compile(r"^[a-z0-9][a-z0-9._-]{1,63}$")
CONFIG_SEGMENT_RE = re.compile(r"^[a-z][a-z0-9_-]*$")
MAX_RESULT_DETAIL_CHARACTERS = 2048
KINDS = {"environment", "correctness", "serving", "performance", "training", "eval", "soak"}
BACKENDS = {"cpu", "cuda", "rocm", "vulkan", "metal"}
VERDICTS = {"passed", "failed"}
RESULT_STATUSES = {"passed", "failed", "skipped"}
ARTIFACT_LOCATIONS = {"local_ignored", "external"}

TOP_LEVEL_KEYS = {
    "schema_version",
    "receipt_id",
    "created_at_utc",
    "source",
    "qualification",
    "environment",
    "model",
    "workload",
    "effective_config",
    "results",
    "metrics",
    "artifacts",
    "unsupported",
    "notes",
}
SOURCE_KEYS = {"tree_hash_format", "tree_hash", "git_commit", "git_worktree_clean"}
QUALIFICATION_KEYS = {
    "kind",
    "backend",
    "profile",
    "verdict",
    "started_at_utc",
    "finished_at_utc",
    "duration_seconds",
    "command",
}
ENVIRONMENT_REQUIRED_KEYS = {"host_id", "os", "device", "runtime", "compiler"}
ENVIRONMENT_KEYS = ENVIRONMENT_REQUIRED_KEYS | {"platform"}
OS_KEYS = {"name", "version", "kernel", "architecture"}
PLATFORM_KEYS = {"kind", "capabilities", "details", "observations"}
WSL2_CAPABILITY_KEYS = {
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
WSL2_OBSERVATION_KEYS = {"host_temperatures", "gpu_temperature"}
MACOS_CAPABILITY_KEYS = {
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
MACOS_OBSERVATION_KEYS = {
    "hardware_identity",
    "metal_runtime",
    "filesystem",
    "unified_memory",
    "memory_pressure",
    "thermal_pressure",
    "host_temperature",
    "gpu_temperature",
}
MACOS_HARDWARE_KEYS = {
    "machine_name",
    "machine_model",
    "chip_type",
    "cpu_brand",
    "gpu_core_count",
    "physical_memory_bytes",
    "kernel_build",
}
MACOS_METAL_RUNTIME_KEYS = {
    "name",
    "has_unified_memory",
    "max_buffer_length_bytes",
    "recommended_max_working_set_bytes",
    "current_allocated_bytes",
}
MACOS_FILESYSTEM_KEYS = {
    "root",
    "source",
    "fstype",
    "mount_point",
    "atomic_replace",
    "full_file_sync",
    "directory_fsync",
    "hardlink",
    "symlink",
    "case_sensitive",
}
MACOS_UNIFIED_MEMORY_KEYS = {
    "total_bytes",
    "swap_total_bytes",
    "swap_used_bytes",
    "swap_encrypted",
}
MACOS_MEMORY_PRESSURE_KEYS = {
    "free_percent",
    "page_size_bytes",
    "pages_free",
    "pages_active",
    "pages_inactive",
    "pages_wired_down",
    "pages_occupied_by_compressor",
    "pageins",
    "pageouts",
    "swapins",
    "swapouts",
}
MACOS_THERMAL_PRESSURE_KEYS = {
    "thermal_warning",
    "performance_warning",
    "cpu_power_status",
}
HOST_TEMPERATURE_KEYS = {"source", "name", "temperature_millicelsius"}
GPU_TEMPERATURE_KEYS = {"source", "device_uuid", "temperature_millicelsius"}
HOST_TEMPERATURE_SOURCES = {"linux_hwmon", "windows_formatted_thermal_zone"}
DEVICE_REQUIRED_KEYS = {
    "name",
    "architecture",
    "memory_bytes",
    "unified_memory",
    "driver",
}
DEVICE_OPTIONAL_KEYS = {
    "logical_index",
    "device_uuid",
    "pci_bus_id",
    "compute_capability",
    "compute_units",
    "memory_available_bytes",
}
DEVICE_KEYS = DEVICE_REQUIRED_KEYS | DEVICE_OPTIONAL_KEYS
MODEL_KEYS = {"id", "path", "weight_files", "config_hash", "tokenizer_hash", "chat_template_hash"}
WEIGHT_KEYS = {"path", "sha256", "bytes"}
WORKLOAD_KEYS = {"id", "sha256", "seed", "parameters"}
RESULT_KEYS = {"id", "required", "status", "duration_seconds", "metrics", "details"}
METRIC_KEYS = {"name", "value", "unit", "aggregation", "lower_is_better"}
ARTIFACT_KEYS = {"kind", "location", "path", "sha256", "bytes"}


class ReceiptLoadError(RuntimeError):
    pass


def atomic_write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = (json.dumps(value, indent=2, sort_keys=True) + "\n").encode("utf-8")
    temp_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            prefix=f".{path.name}.",
            suffix=".tmp",
            dir=path.parent,
            delete=False,
        ) as handle:
            temp_path = Path(handle.name)
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp_path, path)
        temp_path = None
        directory_fd = os.open(path.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        if temp_path is not None:
            temp_path.unlink(missing_ok=True)


def load_receipt(path: Path) -> dict[str, Any]:
    try:
        value = strict_json_loads(path.read_bytes())
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, StrictJSONError) as exc:
        raise ReceiptLoadError(f"cannot load {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ReceiptLoadError(f"{path}: receipt must be a JSON object")
    return value


def _is_number(value: Any) -> bool:
    try:
        return (
            isinstance(value, (int, float))
            and not isinstance(value, bool)
            and math.isfinite(value)
        )
    except OverflowError:
        return False


def _check_exact_keys(errors: list[str], value: Any, expected: set[str], context: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        errors.append(f"{context} must be an object")
        return {}
    missing = sorted(expected - value.keys())
    unknown = sorted(value.keys() - expected)
    if missing:
        errors.append(f"{context} missing keys: {', '.join(missing)}")
    if unknown:
        errors.append(f"{context} has unknown keys: {', '.join(unknown)}")
    return value


def _check_required_keys(
    errors: list[str],
    value: Any,
    required: set[str],
    allowed: set[str],
    context: str,
) -> dict[str, Any]:
    if not isinstance(value, dict):
        errors.append(f"{context} must be an object")
        return {}
    missing = sorted(required - value.keys())
    unknown = sorted(value.keys() - allowed)
    if missing:
        errors.append(f"{context} missing keys: {', '.join(missing)}")
    if unknown:
        errors.append(f"{context} has unknown keys: {', '.join(unknown)}")
    return value


def _check_string(errors: list[str], value: Any, context: str, *, allow_empty: bool = False) -> str:
    if not isinstance(value, str) or (not allow_empty and not value):
        errors.append(f"{context} must be a {'string' if allow_empty else 'non-empty string'}")
        return ""
    return value


def _check_bool(errors: list[str], value: Any, context: str) -> bool | None:
    if not isinstance(value, bool):
        errors.append(f"{context} must be a boolean")
        return None
    return value


def _check_nonnegative_number(errors: list[str], value: Any, context: str) -> float | None:
    if not _is_number(value) or value < 0:
        errors.append(f"{context} must be a finite non-negative number")
        return None
    return float(value)


def _check_positive_int(errors: list[str], value: Any, context: str) -> int | None:
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        errors.append(f"{context} must be a positive integer")
        return None
    return value


def _check_sha256(errors: list[str], value: Any, context: str) -> str:
    if not isinstance(value, str) or not SHA256_RE.fullmatch(value):
        errors.append(f"{context} must match sha256:<64 lowercase hex characters>")
        return ""
    return value


def _parse_timestamp(errors: list[str], value: Any, context: str) -> datetime | None:
    text = _check_string(errors, value, context)
    if not text or not text.endswith("Z"):
        if text:
            errors.append(f"{context} must be an ISO-8601 UTC timestamp ending in Z")
        return None
    try:
        parsed = datetime.fromisoformat(text[:-1] + "+00:00")
    except ValueError:
        errors.append(f"{context} must be a valid ISO-8601 UTC timestamp")
        return None
    if parsed.utcoffset() != timezone.utc.utcoffset(parsed):
        errors.append(f"{context} must use UTC")
        return None
    return parsed


def _check_string_map(errors: list[str], value: Any, context: str) -> None:
    if not isinstance(value, dict) or not value:
        errors.append(f"{context} must be a non-empty object")
        return
    for key, item in value.items():
        if not isinstance(key, str) or not key:
            errors.append(f"{context} keys must be non-empty strings")
        _check_string(errors, item, f"{context}.{key}")


def _check_temperature(
    errors: list[str],
    value: Any,
    context: str,
    *,
    minimum: int = -50_000,
    maximum: int = 200_000,
) -> None:
    if (
        not isinstance(value, int)
        or isinstance(value, bool)
        or not minimum <= value <= maximum
    ):
        errors.append(
            f"{context} must be an integer from {minimum} through {maximum}"
        )


def _validate_wsl2_observations(
    errors: list[str],
    value: Any,
    capabilities: dict[str, Any],
    device: dict[str, Any],
) -> None:
    context = "receipt.environment.platform.observations"
    observations = _check_exact_keys(
        errors,
        value,
        WSL2_OBSERVATION_KEYS,
        context,
    )
    host_temperatures = observations.get("host_temperatures")
    if not isinstance(host_temperatures, list):
        errors.append(f"{context}.host_temperatures must be an array")
        host_temperatures = []
    names: set[tuple[str, str]] = set()
    for index, raw_temperature in enumerate(host_temperatures):
        item_context = f"{context}.host_temperatures[{index}]"
        temperature = _check_exact_keys(
            errors,
            raw_temperature,
            HOST_TEMPERATURE_KEYS,
            item_context,
        )
        source = temperature.get("source")
        if source not in HOST_TEMPERATURE_SOURCES:
            errors.append(
                f"{item_context}.source must be one of "
                f"{sorted(HOST_TEMPERATURE_SOURCES)}"
            )
        name = _check_string(errors, temperature.get("name"), f"{item_context}.name")
        if name and not name.strip():
            errors.append(f"{item_context}.name must contain a non-whitespace character")
        identity = (str(source), name)
        if identity in names:
            errors.append(
                f"{context}.host_temperatures contains duplicate source/name "
                f"{identity!r}"
            )
        names.add(identity)
        _check_temperature(
            errors,
            temperature.get("temperature_millicelsius"),
            f"{item_context}.temperature_millicelsius",
        )

    gpu_value = observations.get("gpu_temperature")
    if gpu_value is not None:
        gpu_context = f"{context}.gpu_temperature"
        gpu_temperature = _check_exact_keys(
            errors,
            gpu_value,
            GPU_TEMPERATURE_KEYS,
            gpu_context,
        )
        if gpu_temperature.get("source") != "nvml":
            errors.append(f"{gpu_context}.source must be 'nvml'")
        device_uuid = _check_string(
            errors,
            gpu_temperature.get("device_uuid"),
            f"{gpu_context}.device_uuid",
        )
        selected_uuid = device.get("device_uuid")
        if not isinstance(selected_uuid, str) or not selected_uuid:
            errors.append(
                f"{gpu_context} requires a selected device UUID"
            )
        elif device_uuid != selected_uuid:
            errors.append(
                f"{gpu_context}.device_uuid must match the selected device"
            )
        _check_temperature(
            errors,
            gpu_temperature.get("temperature_millicelsius"),
            f"{gpu_context}.temperature_millicelsius",
            minimum=1_000,
            maximum=150_000,
        )

    host_available = capabilities.get("host_temperature") == "available"
    if host_available != bool(host_temperatures):
        errors.append(
            "receipt.environment.platform host_temperature capability and "
            "observations disagree"
        )
    gpu_available = capabilities.get("gpu_temperature") == "available"
    if gpu_available != (gpu_value is not None):
        errors.append(
            "receipt.environment.platform gpu_temperature capability and "
            "observations disagree"
        )
    if gpu_available and capabilities.get("nvml") != "available":
        errors.append(
            "receipt.environment.platform gpu_temperature requires NVML"
        )


def _validate_macos_observations(
    errors: list[str],
    value: Any,
    capabilities: dict[str, Any],
    device: dict[str, Any],
) -> None:
    context = "receipt.environment.platform.observations"
    observations = _check_exact_keys(
        errors,
        value,
        MACOS_OBSERVATION_KEYS,
        context,
    )

    hardware_value = observations.get("hardware_identity")
    if hardware_value is not None:
        hardware = _check_exact_keys(
            errors,
            hardware_value,
            MACOS_HARDWARE_KEYS,
            f"{context}.hardware_identity",
        )
        for key in (
            "machine_name",
            "machine_model",
            "chip_type",
            "cpu_brand",
            "kernel_build",
        ):
            _check_string(
                errors,
                hardware.get(key),
                f"{context}.hardware_identity.{key}",
            )
        _check_positive_int(
            errors,
            hardware.get("gpu_core_count"),
            f"{context}.hardware_identity.gpu_core_count",
        )
        _check_positive_int(
            errors,
            hardware.get("physical_memory_bytes"),
            f"{context}.hardware_identity.physical_memory_bytes",
        )
        if hardware.get("chip_type") != device.get("name"):
            errors.append(
                f"{context}.hardware_identity.chip_type must match the selected device"
            )
        if hardware.get("gpu_core_count") != device.get("compute_units"):
            errors.append(
                f"{context}.hardware_identity.gpu_core_count must match the selected device"
            )
        if hardware.get("physical_memory_bytes") != device.get("memory_bytes"):
            errors.append(
                f"{context}.hardware_identity.physical_memory_bytes must match "
                "the selected device"
            )

    metal_value = observations.get("metal_runtime")
    if metal_value is not None:
        metal = _check_exact_keys(
            errors,
            metal_value,
            MACOS_METAL_RUNTIME_KEYS,
            f"{context}.metal_runtime",
        )
        _check_string(errors, metal.get("name"), f"{context}.metal_runtime.name")
        unified = _check_bool(
            errors,
            metal.get("has_unified_memory"),
            f"{context}.metal_runtime.has_unified_memory",
        )
        if unified is not True:
            errors.append(f"{context}.metal_runtime.has_unified_memory must be true")
        for key in (
            "max_buffer_length_bytes",
            "recommended_max_working_set_bytes",
        ):
            _check_positive_int(
                errors,
                metal.get(key),
                f"{context}.metal_runtime.{key}",
            )
        _check_nonnegative_number(
            errors,
            metal.get("current_allocated_bytes"),
            f"{context}.metal_runtime.current_allocated_bytes",
        )
        if metal.get("name") != device.get("name"):
            errors.append(f"{context}.metal_runtime.name must match the selected device")

    filesystem_value = observations.get("filesystem")
    if filesystem_value is not None:
        filesystem = _check_exact_keys(
            errors,
            filesystem_value,
            MACOS_FILESYSTEM_KEYS,
            f"{context}.filesystem",
        )
        for key in ("root", "source", "fstype", "mount_point"):
            _check_string(
                errors,
                filesystem.get(key),
                f"{context}.filesystem.{key}",
            )
        for key in (
            "atomic_replace",
            "full_file_sync",
            "directory_fsync",
            "hardlink",
            "symlink",
        ):
            if _check_bool(
                errors,
                filesystem.get(key),
                f"{context}.filesystem.{key}",
            ) is not True:
                errors.append(f"{context}.filesystem.{key} must be true")
        _check_bool(
            errors,
            filesystem.get("case_sensitive"),
            f"{context}.filesystem.case_sensitive",
        )

    unified_value = observations.get("unified_memory")
    if unified_value is not None:
        unified_memory = _check_exact_keys(
            errors,
            unified_value,
            MACOS_UNIFIED_MEMORY_KEYS,
            f"{context}.unified_memory",
        )
        _check_positive_int(
            errors,
            unified_memory.get("total_bytes"),
            f"{context}.unified_memory.total_bytes",
        )
        for key in ("swap_total_bytes", "swap_used_bytes"):
            value = unified_memory.get(key)
            if (
                not isinstance(value, int)
                or isinstance(value, bool)
                or value < 0
            ):
                errors.append(
                    f"{context}.unified_memory.{key} must be a non-negative integer"
                )
        _check_bool(
            errors,
            unified_memory.get("swap_encrypted"),
            f"{context}.unified_memory.swap_encrypted",
        )
        if unified_memory.get("total_bytes") != device.get("memory_bytes"):
            errors.append(
                f"{context}.unified_memory.total_bytes must match the selected device"
            )
        if (
            isinstance(unified_memory.get("swap_total_bytes"), int)
            and isinstance(unified_memory.get("swap_used_bytes"), int)
            and unified_memory["swap_used_bytes"] > unified_memory["swap_total_bytes"]
        ):
            errors.append(
                f"{context}.unified_memory.swap_used_bytes exceeds swap_total_bytes"
            )

    pressure_value = observations.get("memory_pressure")
    if pressure_value is not None:
        pressure = _check_exact_keys(
            errors,
            pressure_value,
            MACOS_MEMORY_PRESSURE_KEYS,
            f"{context}.memory_pressure",
        )
        for key in MACOS_MEMORY_PRESSURE_KEYS:
            value = pressure.get(key)
            maximum = 100 if key == "free_percent" else None
            if (
                not isinstance(value, int)
                or isinstance(value, bool)
                or value < 0
                or (maximum is not None and value > maximum)
                or (key == "page_size_bytes" and value == 0)
            ):
                bound = " from 0 through 100" if maximum is not None else ""
                errors.append(
                    f"{context}.memory_pressure.{key} must be an integer{bound}"
                )

    thermal_value = observations.get("thermal_pressure")
    if thermal_value is not None:
        thermal = _check_exact_keys(
            errors,
            thermal_value,
            MACOS_THERMAL_PRESSURE_KEYS,
            f"{context}.thermal_pressure",
        )
        for key in MACOS_THERMAL_PRESSURE_KEYS:
            _check_string(
                errors,
                thermal.get(key),
                f"{context}.thermal_pressure.{key}",
            )

    observation_pairs = {
        "apple_hardware_identity": hardware_value,
        "metal_runtime_identity": metal_value,
        "filesystem_semantics": filesystem_value,
        "unified_memory_accounting": unified_value,
        "memory_pressure": pressure_value,
        "thermal_pressure": thermal_value,
    }
    for capability, observation in observation_pairs.items():
        if (capabilities.get(capability) == "available") != (
            observation is not None
        ):
            errors.append(
                f"receipt.environment.platform {capability} capability and "
                "observations disagree"
            )
    for temperature in ("host_temperature", "gpu_temperature"):
        if observations.get(temperature) is not None:
            errors.append(f"{context}.{temperature} must be null on macOS")
        if capabilities.get(temperature) != "unavailable":
            errors.append(
                f"receipt.environment.platform {temperature} must be unavailable "
                "when its observation is null"
            )


def _validate_config(errors: list[str], value: Any, context: str) -> None:
    if not isinstance(value, dict):
        errors.append(f"{context} must be an object")
        return
    for key, item in value.items():
        if not isinstance(key, str) or not CONFIG_SEGMENT_RE.fullmatch(key):
            errors.append(f"{context} key {key!r} must be a dot-path-compatible segment")
            continue
        item_context = f"{context}.{key}"
        if isinstance(item, dict):
            _validate_config(errors, item, item_context)
        elif item is None or isinstance(item, (str, bool)) or _is_number(item):
            continue
        else:
            errors.append(f"{item_context} must be a finite JSON scalar or nested object")


def _validate_metric(errors: list[str], value: Any, context: str) -> str:
    metric = _check_exact_keys(errors, value, METRIC_KEYS, context)
    name = _check_string(errors, metric.get("name"), f"{context}.name")
    if name and not METRIC_NAME_RE.fullmatch(name):
        errors.append(f"{context}.name has invalid metric syntax")
    if not _is_number(metric.get("value")):
        errors.append(f"{context}.value must be finite numeric")
    _check_string(errors, metric.get("unit"), f"{context}.unit")
    _check_string(errors, metric.get("aggregation"), f"{context}.aggregation")
    _check_bool(errors, metric.get("lower_is_better"), f"{context}.lower_is_better")
    return name


def _validate_metrics(errors: list[str], value: Any, context: str) -> None:
    if not isinstance(value, list):
        errors.append(f"{context} must be an array")
        return
    names: set[str] = set()
    for index, metric in enumerate(value):
        name = _validate_metric(errors, metric, f"{context}[{index}]")
        if name in names:
            errors.append(f"{context} contains duplicate metric name {name!r}")
        names.add(name)


def _validate_model(errors: list[str], value: Any, context: str) -> None:
    model = _check_exact_keys(errors, value, MODEL_KEYS, context)
    _check_string(errors, model.get("id"), f"{context}.id")
    _check_string(errors, model.get("path"), f"{context}.path")
    for key in ("config_hash", "tokenizer_hash"):
        _check_sha256(errors, model.get(key), f"{context}.{key}")
    template_hash = model.get("chat_template_hash")
    if template_hash is not None:
        _check_sha256(errors, template_hash, f"{context}.chat_template_hash")

    weights = model.get("weight_files")
    if not isinstance(weights, list) or not weights:
        errors.append(f"{context}.weight_files must be a non-empty array")
        return
    paths: set[str] = set()
    for index, raw_weight in enumerate(weights):
        weight_context = f"{context}.weight_files[{index}]"
        weight = _check_exact_keys(errors, raw_weight, WEIGHT_KEYS, weight_context)
        path = _check_string(errors, weight.get("path"), f"{weight_context}.path")
        if path in paths:
            errors.append(f"{context}.weight_files contains duplicate path {path!r}")
        paths.add(path)
        _check_sha256(errors, weight.get("sha256"), f"{weight_context}.sha256")
        _check_positive_int(errors, weight.get("bytes"), f"{weight_context}.bytes")


def _validate_workload(errors: list[str], value: Any, context: str) -> None:
    workload = _check_exact_keys(errors, value, WORKLOAD_KEYS, context)
    _check_string(errors, workload.get("id"), f"{context}.id")
    _check_sha256(errors, workload.get("sha256"), f"{context}.sha256")
    seed = workload.get("seed")
    if seed is not None and (
        not isinstance(seed, int) or isinstance(seed, bool) or seed < 0
    ):
        errors.append(f"{context}.seed must be null or a non-negative integer")
    if not isinstance(workload.get("parameters"), dict):
        errors.append(f"{context}.parameters must be an object")


def validate_receipt(
    receipt: dict[str, Any],
    *,
    root: Path = ROOT,
    require_current_source: bool = False,
    require_local_artifacts: bool = False,
) -> list[str]:
    errors: list[str] = []
    top = _check_exact_keys(errors, receipt, TOP_LEVEL_KEYS, "receipt")
    if top.get("schema_version") != SCHEMA_VERSION:
        errors.append(f"receipt.schema_version must be {SCHEMA_VERSION}")
    receipt_id = _check_string(errors, top.get("receipt_id"), "receipt.receipt_id")
    if receipt_id and not ID_RE.fullmatch(receipt_id):
        errors.append("receipt.receipt_id has invalid syntax")
    created_at = _parse_timestamp(errors, top.get("created_at_utc"), "receipt.created_at_utc")

    source = _check_exact_keys(errors, top.get("source"), SOURCE_KEYS, "receipt.source")
    if source.get("tree_hash_format") != HASH_FORMAT:
        errors.append(f"receipt.source.tree_hash_format must be {HASH_FORMAT!r}")
    tree_hash = _check_sha256(errors, source.get("tree_hash"), "receipt.source.tree_hash")
    commit = source.get("git_commit")
    if not isinstance(commit, str) or not COMMIT_RE.fullmatch(commit):
        errors.append("receipt.source.git_commit must be a lowercase 40-character commit")
    clean = _check_bool(
        errors, source.get("git_worktree_clean"), "receipt.source.git_worktree_clean"
    )

    qualification = _check_exact_keys(
        errors, top.get("qualification"), QUALIFICATION_KEYS, "receipt.qualification"
    )
    kind = qualification.get("kind")
    if kind not in KINDS:
        errors.append(f"receipt.qualification.kind must be one of {sorted(KINDS)}")
    backend = qualification.get("backend")
    if backend not in BACKENDS:
        errors.append(f"receipt.qualification.backend must be one of {sorted(BACKENDS)}")
    _check_string(errors, qualification.get("profile"), "receipt.qualification.profile")
    verdict = qualification.get("verdict")
    if verdict not in VERDICTS:
        errors.append(f"receipt.qualification.verdict must be one of {sorted(VERDICTS)}")
    started_at = _parse_timestamp(
        errors, qualification.get("started_at_utc"), "receipt.qualification.started_at_utc"
    )
    finished_at = _parse_timestamp(
        errors, qualification.get("finished_at_utc"), "receipt.qualification.finished_at_utc"
    )
    duration = _check_nonnegative_number(
        errors, qualification.get("duration_seconds"), "receipt.qualification.duration_seconds"
    )
    command = qualification.get("command")
    if not isinstance(command, list) or not command:
        errors.append("receipt.qualification.command must be a non-empty argv array")
    else:
        for index, item in enumerate(command):
            _check_string(errors, item, f"receipt.qualification.command[{index}]")

    if started_at is not None and finished_at is not None:
        elapsed = (finished_at - started_at).total_seconds()
        if elapsed < 0:
            errors.append("receipt.qualification.finished_at_utc precedes started_at_utc")
        elif duration is not None and abs(duration - elapsed) > 1.0:
            errors.append(
                "receipt.qualification.duration_seconds differs from timestamps by more than 1 second"
            )
        if created_at is not None and created_at < finished_at:
            errors.append("receipt.created_at_utc precedes qualification completion")

    environment = _check_required_keys(
        errors,
        top.get("environment"),
        ENVIRONMENT_REQUIRED_KEYS,
        ENVIRONMENT_KEYS,
        "receipt.environment",
    )
    host_id = _check_string(errors, environment.get("host_id"), "receipt.environment.host_id")
    if host_id and not HOST_ID_RE.fullmatch(host_id):
        errors.append("receipt.environment.host_id has invalid syntax")
    os_value = _check_exact_keys(errors, environment.get("os"), OS_KEYS, "receipt.environment.os")
    for key in OS_KEYS:
        _check_string(errors, os_value.get(key), f"receipt.environment.os.{key}")
    device = _check_required_keys(
        errors,
        environment.get("device"),
        DEVICE_REQUIRED_KEYS,
        DEVICE_KEYS,
        "receipt.environment.device",
    )
    for key in ("name", "architecture", "driver"):
        _check_string(errors, device.get(key), f"receipt.environment.device.{key}")
    memory_bytes = device.get("memory_bytes")
    if memory_bytes is not None:
        _check_positive_int(errors, memory_bytes, "receipt.environment.device.memory_bytes")
    _check_bool(
        errors, device.get("unified_memory"), "receipt.environment.device.unified_memory"
    )
    logical_index = device.get("logical_index")
    if logical_index is not None and (
        not isinstance(logical_index, int)
        or isinstance(logical_index, bool)
        or logical_index < 0
    ):
        errors.append(
            "receipt.environment.device.logical_index must be a non-negative integer or null"
        )
    for key in ("device_uuid", "pci_bus_id", "compute_capability"):
        value = device.get(key)
        if value is not None:
            _check_string(errors, value, f"receipt.environment.device.{key}")
    compute_units = device.get("compute_units")
    if compute_units is not None:
        _check_positive_int(
            errors, compute_units, "receipt.environment.device.compute_units"
        )
    memory_available = device.get("memory_available_bytes")
    if memory_available is not None:
        if (
            not isinstance(memory_available, int)
            or isinstance(memory_available, bool)
            or memory_available < 0
        ):
            errors.append(
                "receipt.environment.device.memory_available_bytes must be a "
                "non-negative integer or null"
            )
        elif isinstance(memory_bytes, int) and memory_available > memory_bytes:
            errors.append(
                "receipt.environment.device.memory_available_bytes exceeds memory_bytes"
            )
    _check_string_map(errors, environment.get("runtime"), "receipt.environment.runtime")
    _check_string_map(errors, environment.get("compiler"), "receipt.environment.compiler")
    if "platform" in environment:
        platform_value = environment.get("platform")
        platform_object = _check_exact_keys(
            errors,
            platform_value,
            PLATFORM_KEYS,
            "receipt.environment.platform",
        )
        platform_kind = platform_object.get("kind")
        if platform_kind not in {"wsl2", "macos"}:
            errors.append(
                "receipt.environment.platform.kind must be 'wsl2' or 'macos'"
            )
        capability_keys = (
            MACOS_CAPABILITY_KEYS
            if platform_kind == "macos"
            else WSL2_CAPABILITY_KEYS
        )
        capabilities = _check_exact_keys(
            errors,
            platform_object.get("capabilities"),
            capability_keys,
            "receipt.environment.platform.capabilities",
        )
        for key, status in capabilities.items():
            if status not in CAPABILITY_STATUSES:
                errors.append(
                    f"receipt.environment.platform.capabilities.{key} must be one of "
                    f"{sorted(CAPABILITY_STATUSES)}"
                )
        _check_string_map(
            errors,
            platform_object.get("details"),
            "receipt.environment.platform.details",
        )
        if platform_kind == "macos":
            _validate_macos_observations(
                errors,
                platform_object.get("observations"),
                capabilities,
                device,
            )
        else:
            _validate_wsl2_observations(
                errors,
                platform_object.get("observations"),
                capabilities,
                device,
            )

    model = top.get("model")
    workload = top.get("workload")
    if model is not None:
        _validate_model(errors, model, "receipt.model")
    if workload is not None:
        _validate_workload(errors, workload, "receipt.workload")
    if (
        kind in {"serving", "performance", "training", "eval", "soak"}
        and verdict == "passed"
        and model is None
    ):
        errors.append(
            f"receipt.model is required for passed qualification kind {kind!r}"
        )
    if kind in KINDS - {"environment"}:
        if workload is None:
            errors.append(f"receipt.workload is required for qualification kind {kind!r}")
    _validate_config(errors, top.get("effective_config"), "receipt.effective_config")

    results = top.get("results")
    required_failures = 0
    if not isinstance(results, list) or not results:
        errors.append("receipt.results must be a non-empty array")
    else:
        result_ids: set[str] = set()
        for index, raw_result in enumerate(results):
            context = f"receipt.results[{index}]"
            result = _check_exact_keys(errors, raw_result, RESULT_KEYS, context)
            result_id = _check_string(errors, result.get("id"), f"{context}.id")
            if result_id and not RESULT_ID_RE.fullmatch(result_id):
                errors.append(f"{context}.id has invalid syntax")
            if result_id in result_ids:
                errors.append(f"receipt.results contains duplicate id {result_id!r}")
            result_ids.add(result_id)
            required = _check_bool(errors, result.get("required"), f"{context}.required")
            status = result.get("status")
            if status not in RESULT_STATUSES:
                errors.append(f"{context}.status must be one of {sorted(RESULT_STATUSES)}")
            if required and status != "passed":
                required_failures += 1
            _check_nonnegative_number(errors, result.get("duration_seconds"), f"{context}.duration_seconds")
            _validate_metrics(errors, result.get("metrics"), f"{context}.metrics")
            details = result.get("details")
            if details is not None:
                checked_details = _check_string(
                    errors, details, f"{context}.details", allow_empty=True
                )
                if (
                    checked_details is not None
                    and len(checked_details) > MAX_RESULT_DETAIL_CHARACTERS
                ):
                    errors.append(
                        f"{context}.details must be at most "
                        f"{MAX_RESULT_DETAIL_CHARACTERS} characters"
                    )

    _validate_metrics(errors, top.get("metrics"), "receipt.metrics")

    artifacts = top.get("artifacts")
    if not isinstance(artifacts, list):
        errors.append("receipt.artifacts must be an array")
    else:
        artifact_keys: set[tuple[str, str]] = set()
        for index, raw_artifact in enumerate(artifacts):
            context = f"receipt.artifacts[{index}]"
            artifact = _check_exact_keys(errors, raw_artifact, ARTIFACT_KEYS, context)
            kind_value = _check_string(errors, artifact.get("kind"), f"{context}.kind")
            location = artifact.get("location")
            if location not in ARTIFACT_LOCATIONS:
                errors.append(f"{context}.location must be one of {sorted(ARTIFACT_LOCATIONS)}")
            path_value = _check_string(errors, artifact.get("path"), f"{context}.path")
            if (kind_value, path_value) in artifact_keys:
                errors.append(f"receipt.artifacts contains duplicate kind/path {kind_value!r}/{path_value!r}")
            artifact_keys.add((kind_value, path_value))
            expected_hash = _check_sha256(errors, artifact.get("sha256"), f"{context}.sha256")
            expected_bytes = artifact.get("bytes")
            if not isinstance(expected_bytes, int) or isinstance(expected_bytes, bool) or expected_bytes < 0:
                errors.append(f"{context}.bytes must be a non-negative integer")
            if require_local_artifacts and location == "local_ignored" and path_value:
                artifact_path = Path(path_value)
                if not artifact_path.is_absolute():
                    artifact_path = root / artifact_path
                try:
                    resolved = artifact_path.resolve(strict=True)
                    ignored_root = (root / ".qualification").resolve()
                    resolved.relative_to(ignored_root)
                except (OSError, ValueError):
                    errors.append(f"{context}.path must exist under .qualification for local validation")
                else:
                    content = resolved.read_bytes()
                    observed_hash = f"sha256:{hashlib.sha256(content).hexdigest()}"
                    if expected_hash and observed_hash != expected_hash:
                        errors.append(f"{context}.sha256 does not match local artifact")
                    if isinstance(expected_bytes, int) and len(content) != expected_bytes:
                        errors.append(f"{context}.bytes does not match local artifact")

    for key in ("unsupported", "notes"):
        value = top.get(key)
        if not isinstance(value, list):
            errors.append(f"receipt.{key} must be an array")
        else:
            for index, item in enumerate(value):
                _check_string(errors, item, f"receipt.{key}[{index}]")

    if verdict == "passed":
        if clean is not True:
            errors.append("a passed receipt requires receipt.source.git_worktree_clean=true")
        if required_failures:
            errors.append("a passed receipt cannot contain failed or skipped required results")
    elif verdict == "failed" and required_failures == 0:
        errors.append("a failed receipt must contain at least one failed or skipped required result")

    if require_current_source and tree_hash:
        try:
            current_hash, _ = source_tree_hash(root)
        except SourceTreeHashError as exc:
            errors.append(f"cannot compute current source tree hash: {exc}")
        else:
            if current_hash != tree_hash:
                errors.append(
                    f"receipt.source.tree_hash is {tree_hash}, current source tree is {current_hash}"
                )
    return errors


def _git_commit_exists(root: Path, commit: str) -> bool:
    if not COMMIT_RE.fullmatch(commit):
        return False
    return subprocess.run(
        ["git", "cat-file", "-e", f"{commit}^{{commit}}"],
        cwd=root,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=False,
    ).returncode == 0


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("receipts", nargs="+", type=Path)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--require-current-source", action="store_true")
    parser.add_argument("--require-local-artifacts", action="store_true")
    parser.add_argument("--require-known-commit", action="store_true")
    parser.add_argument("--json", action="store_true", dest="json_output")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(sys.argv[1:] if argv is None else argv)
    root = args.root.resolve()
    report: list[dict[str, Any]] = []
    failed = False
    for receipt_path in args.receipts:
        path = receipt_path if receipt_path.is_absolute() else root / receipt_path
        try:
            receipt = load_receipt(path)
        except ReceiptLoadError as exc:
            errors = [str(exc)]
        else:
            errors = validate_receipt(
                receipt,
                root=root,
                require_current_source=args.require_current_source,
                require_local_artifacts=args.require_local_artifacts,
            )
            commit = receipt.get("source", {}).get("git_commit")
            if args.require_known_commit and isinstance(commit, str) and not _git_commit_exists(root, commit):
                errors.append(f"receipt.source.git_commit does not exist locally: {commit}")
        failed = failed or bool(errors)
        report.append({"path": str(receipt_path), "ok": not errors, "errors": errors})

    if args.json_output:
        print(json.dumps({"ok": not failed, "receipts": report}, indent=2, sort_keys=True))
    else:
        for item in report:
            if item["ok"]:
                print(f"OK {item['path']}")
            else:
                print(f"FAILED {item['path']}", file=sys.stderr)
                for error in item["errors"]:
                    print(f"  - {error}", file=sys.stderr)
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
