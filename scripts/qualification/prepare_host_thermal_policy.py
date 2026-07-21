#!/usr/bin/env python3
"""Inventory labeled Linux temperature sensors and publish a closed host policy."""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import tempfile
from pathlib import Path
from typing import Any, Sequence

import host_thermal_guard as thermal
import host_thermal_policy as policy


INVENTORY_SCHEMA = "kiln.host-thermal-sensor-inventory.v1"


class PreparationError(RuntimeError):
    """Raised when a policy cannot be prepared safely."""


def _read_nonempty(path: Path, label: str) -> str:
    try:
        value = path.read_text(encoding="utf-8").strip()
    except OSError as exc:
        raise PreparationError(f"cannot read {label} {path}: {exc}") from exc
    if not value:
        raise PreparationError(f"{label} is empty: {path}")
    return value


def inventory(hwmon_root: Path = Path("/sys/class/hwmon")) -> dict[str, Any]:
    """Return every readable labeled hwmon temperature input in stable order."""

    if not hwmon_root.is_dir():
        raise PreparationError(f"hwmon root is not a directory: {hwmon_root}")
    sensors: list[dict[str, Any]] = []
    selector_counts: dict[tuple[str, str], int] = {}
    for device in sorted(hwmon_root.glob("hwmon*")):
        name_path = device / "name"
        if not name_path.is_file():
            continue
        hwmon_name = _read_nonempty(name_path, "hwmon name")
        for label_path in sorted(device.glob("temp*_label")):
            match = re.fullmatch(r"temp([1-9][0-9]*)_label", label_path.name)
            if match is None or not label_path.is_file():
                continue
            label = _read_nonempty(label_path, "temperature label")
            input_path = label_path.with_name(f"temp{match.group(1)}_input")
            if not input_path.is_file():
                raise PreparationError(
                    f"labeled sensor {hwmon_name}/{label} has no input: {input_path}"
                )
            temperature = thermal.read_hwmon_temperature_millicelsius(
                input_path,
                error_type=PreparationError,
            )
            key = (hwmon_name, label)
            selector_counts[key] = selector_counts.get(key, 0) + 1
            sensors.append(
                {
                    "hwmon_name": hwmon_name,
                    "input_path": str(input_path.absolute()),
                    "label": label,
                    "temperature_millicelsius": temperature,
                }
            )
    for sensor in sensors:
        sensor["selector_resolves_uniquely"] = (
            selector_counts[(sensor["hwmon_name"], sensor["label"])] == 1
        )
    return {
        "schema": INVENTORY_SCHEMA,
        "hwmon_root": str(hwmon_root.absolute()),
        "sensor_count": len(sensors),
        "sensors": sensors,
    }


def build_policy(args: argparse.Namespace) -> tuple[dict[str, Any], Path, int]:
    """Build, validate, and resolve one hard-limit-only policy."""

    if not 1 <= args.limit_millicelsius <= 200_000:
        raise PreparationError("hard limit must be in 1..=200000 millicelsius")
    if not 50 <= args.poll_interval_ms <= 60_000:
        raise PreparationError("poll interval must be in 50..=60000 milliseconds")
    raw = {
        "schema": policy.SCHEMA,
        "id": args.id,
        "sensor": {
            "hwmon_name": args.hwmon_name,
            "label": args.label,
        },
        "limit_millicelsius": args.limit_millicelsius,
        "poll_interval_ms": args.poll_interval_ms,
        "pacing": {"mode": "hard_limit_only"},
        "safe_handoff": {
            "target_millicelsius": args.safe_handoff_target_millicelsius,
            "stable_samples": args.safe_handoff_stable_samples,
            "timeout_seconds": args.safe_handoff_timeout_seconds,
        },
        "phase_settlement_timeout_seconds": args.phase_settlement_timeout_seconds,
    }
    normalized, parsed, _ = policy.validate(raw, error_type=PreparationError)
    input_path = thermal.resolve_hwmon_temperature_input(
        parsed.hwmon_name,
        parsed.label,
        args.hwmon_root,
        error_type=PreparationError,
    )
    temperature = thermal.read_hwmon_temperature_millicelsius(
        input_path,
        error_type=PreparationError,
    )
    if temperature >= parsed.limit_millicelsius:
        raise PreparationError(
            f"refusing policy publication at {temperature} millicelsius; "
            f"hard limit is {parsed.limit_millicelsius}"
        )
    return normalized, input_path, temperature


def cargo_fields(path: Path, hwmon_root: Path) -> tuple[str, str, int, int]:
    """Resolve a content-hashed policy into the Cargo wrapper's guard fields."""

    try:
        raw = json.loads(path.read_bytes())
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise PreparationError(f"cannot read host thermal policy {path}: {exc}") from exc
    if not isinstance(raw, dict) or "content_sha256" not in raw:
        raise PreparationError("Cargo requires a content-hashed host thermal policy")
    normalized, parsed, _ = policy.load(path, error_type=PreparationError)
    if normalized["pacing"] != {"mode": "hard_limit_only"}:
        raise PreparationError("Cargo requires a hard_limit_only host thermal policy")
    for label, value in (
        ("hwmon name", parsed.hwmon_name),
        ("sensor label", parsed.label),
    ):
        if any(character in value for character in ("\0", "\r", "\n")):
            raise PreparationError(f"policy {label} must not contain control characters")
    input_path = thermal.resolve_hwmon_temperature_input(
        parsed.hwmon_name,
        parsed.label,
        hwmon_root,
        error_type=PreparationError,
    )
    temperature = thermal.read_hwmon_temperature_millicelsius(
        input_path,
        error_type=PreparationError,
    )
    if temperature >= parsed.limit_millicelsius:
        raise PreparationError(
            f"refusing Cargo at {temperature} millicelsius; "
            f"thermal limit is {parsed.limit_millicelsius}"
        )
    policy.wait_for_prelaunch_cooldown(
        parsed,
        hwmon_root=hwmon_root,
        error_type=PreparationError,
    )
    return (
        parsed.hwmon_name,
        parsed.label,
        parsed.limit_millicelsius,
        parsed.poll_interval_ms,
    )


def publish_no_clobber(path: Path, value: dict[str, Any]) -> None:
    """Durably publish canonical pretty JSON without replacing any path."""

    parent = path.parent
    if not parent.is_dir():
        raise PreparationError(f"policy output parent is not a directory: {parent}")
    payload = (
        json.dumps(value, allow_nan=False, ensure_ascii=True, indent=2, sort_keys=False)
        + "\n"
    ).encode("ascii")
    temporary_path: Path | None = None
    try:
        descriptor, temporary = tempfile.mkstemp(
            prefix=f".{path.name}.",
            suffix=".tmp",
            dir=parent,
        )
        temporary_path = Path(temporary)
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            os.fchmod(handle.fileno(), 0o644)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary_path, path, follow_symlinks=False)
        except FileExistsError as exc:
            raise PreparationError(f"refusing to replace existing policy: {path}") from exc
        directory_descriptor = os.open(parent, os.O_RDONLY | getattr(os, "O_CLOEXEC", 0))
        try:
            os.fsync(directory_descriptor)
        finally:
            os.close(directory_descriptor)
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--hwmon-root",
        type=Path,
        default=Path("/sys/class/hwmon"),
        help="Linux hwmon class root (default: /sys/class/hwmon)",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser(
        "inventory",
        help="print strict JSON for every readable labeled temperature input",
    )
    create = subparsers.add_parser(
        "create",
        help="resolve one selector and publish a no-clobber hard-limit policy",
    )
    create.add_argument("--id", required=True)
    create.add_argument("--hwmon-name", required=True)
    create.add_argument("--label", required=True)
    create.add_argument("--limit-millicelsius", type=int, required=True)
    create.add_argument("--poll-interval-ms", type=int, default=250)
    create.add_argument(
        "--safe-handoff-target-millicelsius",
        type=int,
        required=True,
    )
    create.add_argument("--safe-handoff-stable-samples", type=int, default=8)
    create.add_argument("--safe-handoff-timeout-seconds", type=float, default=300.0)
    create.add_argument(
        "--phase-settlement-timeout-seconds",
        type=float,
        default=300.0,
    )
    create.add_argument("--output", type=Path, required=True)
    cargo = subparsers.add_parser(
        "cargo-fields",
        help="resolve one content-hashed hard-limit policy for cargo-bounded.sh",
    )
    cargo.add_argument("--policy", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    try:
        args = parse_args(argv)
        if args.command == "inventory":
            result = inventory(args.hwmon_root)
        elif args.command == "cargo-fields":
            for value in cargo_fields(args.policy, args.hwmon_root):
                print(value)
            return 0
        else:
            normalized, input_path, temperature = build_policy(args)
            publish_no_clobber(args.output, normalized)
            result = {
                "policy": normalized,
                "output": str(args.output.absolute()),
                "resolved_input_path": str(input_path.absolute()),
                "temperature_millicelsius": temperature,
            }
        print(
            json.dumps(
                result,
                allow_nan=False,
                ensure_ascii=True,
                separators=(",", ":"),
                sort_keys=True,
            )
        )
        return 0
    except (OSError, PreparationError, thermal.ThermalGuardError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
