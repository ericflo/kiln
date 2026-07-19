"""Closed JSON policy loading and prelaunch cooling for host thermal guards."""

from __future__ import annotations

import hashlib
import json
import math
import re
import time
from pathlib import Path
from typing import Any, Callable

import host_thermal_guard as thermal
from strict_json import loads as strict_json_loads


SCHEMA = "kiln.host-thermal-policy.v1"
POLICY_KEYS = {
    "id",
    "limit_millicelsius",
    "pacing",
    "phase_settlement_timeout_seconds",
    "poll_interval_ms",
    "safe_handoff",
    "schema",
    "sensor",
}
TraceCallback = Callable[..., None]


def _canonical_sha256(value: Any) -> str:
    payload = json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("ascii")
    return f"sha256:{hashlib.sha256(payload).hexdigest()}"


def _object(value: Any, label: str, error_type: type[Exception]) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise error_type(f"{label} must be an object")
    return value


def _exact_keys(
    value: dict[str, Any], expected: set[str], label: str, error_type: type[Exception]
) -> None:
    actual = set(value)
    if actual != expected:
        missing = sorted(expected - actual)
        unexpected = sorted(actual - expected)
        if missing:
            raise error_type(f"{label} missing keys: {', '.join(missing)}")
        raise error_type(f"{label} has unknown keys: {', '.join(unexpected)}")


def _positive_finite(value: Any, label: str, error_type: type[Exception]) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
        or value <= 0
    ):
        raise error_type(f"{label} must be positive and finite")
    return float(value)


def validate(
    value: Any,
    label: str = "host thermal policy",
    *,
    error_type: type[Exception] = thermal.ThermalGuardError,
    cooldown_mode: str = "live_process_safe_handoff",
) -> tuple[dict[str, Any], thermal.HostThermalPolicy, float]:
    """Validate one closed policy document and bind its canonical content hash."""

    value = _object(value, label, error_type)
    has_content_hash = "content_sha256" in value
    _exact_keys(
        value,
        POLICY_KEYS | ({"content_sha256"} if has_content_hash else set()),
        label,
        error_type,
    )
    raw = dict(value)
    recorded_hash = raw.pop("content_sha256", None)
    if raw["schema"] != SCHEMA:
        raise error_type(f"host thermal policy must use schema {SCHEMA}")
    if not isinstance(raw["id"], str) or re.fullmatch(
        r"[a-z0-9][a-z0-9._-]{2,127}", raw["id"]
    ) is None:
        raise error_type("host thermal policy id must be a portable identifier")

    sensor = _object(raw["sensor"], f"{label}.sensor", error_type)
    _exact_keys(sensor, {"hwmon_name", "label"}, f"{label}.sensor", error_type)
    for name in ("hwmon_name", "label"):
        if not isinstance(sensor[name], str) or not sensor[name]:
            raise error_type(f"{label}.sensor.{name} must be non-empty")

    pacing = _object(raw["pacing"], f"{label}.pacing", error_type)
    legacy_hashed_pacing = has_content_hash and "mode" not in pacing
    if legacy_hashed_pacing:
        _exact_keys(
            pacing,
            {"start_millicelsius", "resume_millicelsius"}
            | (
                {"resume_stable_samples"}
                if "resume_stable_samples" in pacing
                else set()
            ),
            f"{label}.pacing",
            error_type,
        )
        pacing_mode = "process_group_stop"
    else:
        pacing_mode = pacing.get("mode")
        if pacing_mode == "process_group_stop":
            _exact_keys(
                pacing,
                {
                    "mode",
                    "start_millicelsius",
                    "resume_millicelsius",
                    "resume_stable_samples",
                },
                f"{label}.pacing",
                error_type,
            )
        elif pacing_mode == "hard_limit_only":
            _exact_keys(pacing, {"mode"}, f"{label}.pacing", error_type)
        else:
            raise error_type(
                f"{label}.pacing.mode must be process_group_stop or hard_limit_only"
            )
    handoff = _object(raw["safe_handoff"], f"{label}.safe_handoff", error_type)
    _exact_keys(
        handoff,
        {"target_millicelsius", "stable_samples", "timeout_seconds"},
        f"{label}.safe_handoff",
        error_type,
    )
    settlement_timeout = _positive_finite(
        raw["phase_settlement_timeout_seconds"],
        f"{label}.phase_settlement_timeout_seconds",
        error_type,
    )
    policy = thermal.HostThermalPolicy(
        hwmon_name=sensor["hwmon_name"],
        label=sensor["label"],
        limit_millicelsius=raw["limit_millicelsius"],
        poll_interval_ms=raw["poll_interval_ms"],
        pacing_start_millicelsius=(
            pacing["start_millicelsius"]
            if pacing_mode == "process_group_stop"
            else None
        ),
        pacing_resume_millicelsius=(
            pacing["resume_millicelsius"]
            if pacing_mode == "process_group_stop"
            else None
        ),
        pacing_resume_stable_samples=(
            pacing.get("resume_stable_samples", 1)
            if pacing_mode == "process_group_stop"
            else 1
        ),
        pacing_timeout_seconds=(
            settlement_timeout if pacing_mode == "process_group_stop" else None
        ),
        cooldown_target_millicelsius=handoff["target_millicelsius"],
        cooldown_stable_samples=handoff["stable_samples"],
        cooldown_timeout_seconds=handoff["timeout_seconds"],
        cooldown_mode=cooldown_mode,
        error_type=error_type,
    )
    content_hash = _canonical_sha256(raw)
    if recorded_hash is not None:
        if not isinstance(recorded_hash, str) or re.fullmatch(
            r"sha256:[0-9a-f]{64}", recorded_hash
        ) is None:
            raise error_type(f"{label}.content_sha256 must be canonical sha256")
        if recorded_hash != content_hash:
            raise error_type(f"{label}.content_sha256 does not match policy content")
    normalized = dict(raw)
    normalized["content_sha256"] = content_hash
    return normalized, policy, settlement_timeout


def load(
    path: Path,
    *,
    error_type: type[Exception] = thermal.ThermalGuardError,
    cooldown_mode: str = "live_process_safe_handoff",
) -> tuple[dict[str, Any], thermal.HostThermalPolicy, float]:
    """Load a non-symlink policy file with strict JSON semantics."""

    if path.is_symlink() or not path.is_file():
        raise error_type(f"host thermal policy is not a regular file: {path}")
    data = path.read_bytes()
    if len(data) > 64 * 1024:
        raise error_type("host thermal policy exceeds 64 KiB")
    try:
        raw = strict_json_loads(data)
    except Exception as exc:
        raise error_type(f"cannot load host thermal policy {path}: {exc}") from exc
    return validate(
        raw,
        error_type=error_type,
        cooldown_mode=cooldown_mode,
    )


def wait_for_prelaunch_cooldown(
    policy: thermal.HostThermalPolicy,
    *,
    hwmon_root: Path = Path("/sys/class/hwmon"),
    trace_callback: TraceCallback | None = None,
    error_type: type[Exception] = thermal.ThermalGuardError,
) -> dict[str, Any]:
    """Require stable cooling after provenance work and before process creation."""

    input_path = thermal.resolve_hwmon_temperature_input(
        policy.hwmon_name,
        policy.label,
        hwmon_root,
        error_type=error_type,
    )
    poll_interval_seconds = policy.poll_interval_ms / 1000.0
    started = time.monotonic()
    deadline = started + policy.cooldown_timeout_seconds
    sample_count = 0
    stable_samples = 0
    start_temperature: int | None = None
    peak_temperature: int | None = None
    end_temperature: int | None = None

    def trace(event: str, **fields: Any) -> None:
        if trace_callback is not None:
            trace_callback(event, **fields)

    trace(
        "host_thermal_prelaunch_cooldown_started",
        scope="host_package_before_process_creation",
        sensor_path=str(input_path),
        target_millicelsius=policy.cooldown_target_millicelsius,
        stable_samples=policy.cooldown_stable_samples,
        timeout_seconds=policy.cooldown_timeout_seconds,
        poll_interval_ms=policy.poll_interval_ms,
    )
    while True:
        temperature = thermal.read_hwmon_temperature_millicelsius(
            input_path,
            error_type=error_type,
        )
        sample_count += 1
        if start_temperature is None:
            start_temperature = temperature
            peak_temperature = temperature
        assert peak_temperature is not None
        peak_temperature = max(peak_temperature, temperature)
        end_temperature = temperature
        stable_samples = (
            stable_samples + 1
            if temperature <= policy.cooldown_target_millicelsius
            else 0
        )
        elapsed = time.monotonic() - started
        if stable_samples >= policy.cooldown_stable_samples:
            evidence = {
                "completed": True,
                "elapsed_seconds": elapsed,
                "poll_interval_ms": policy.poll_interval_ms,
                "sample_count": sample_count,
                "scope": "host_package_before_process_creation",
                "sensor_path": str(input_path),
                "stable_samples_observed": stable_samples,
                "stable_samples_required": policy.cooldown_stable_samples,
                "target_millicelsius": policy.cooldown_target_millicelsius,
                "temperature_end_millicelsius": end_temperature,
                "temperature_peak_millicelsius": peak_temperature,
                "temperature_start_millicelsius": start_temperature,
                "timeout_seconds": policy.cooldown_timeout_seconds,
            }
            trace("host_thermal_prelaunch_cooldown_completed", **evidence)
            return evidence
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            trace(
                "host_thermal_prelaunch_cooldown_timed_out",
                elapsed_seconds=elapsed,
                sample_count=sample_count,
                sensor_path=str(input_path),
                stable_samples_observed=stable_samples,
                temperature_end_millicelsius=end_temperature,
                temperature_peak_millicelsius=peak_temperature,
                temperature_start_millicelsius=start_temperature,
            )
            raise error_type(
                "host thermal pre-launch cooldown did not reach "
                f"{policy.cooldown_target_millicelsius} millicelsius for "
                f"{policy.cooldown_stable_samples} consecutive samples within "
                f"{policy.cooldown_timeout_seconds:.3f} seconds"
            )
        time.sleep(min(poll_interval_seconds, remaining))
