"""Strict reader for controller-authenticated WSL2 thermal pacing events."""

from __future__ import annotations

import math
import os
import stat
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from strict_json import loads as strict_json_loads


PACING_EVENTS_PATH_ENV = "KILN_WSL2_THERMAL_PACING_EVENTS_PATH"
PACING_EVENT_SCHEMA = "kiln.wsl2-thermal-pacing-event.v1"
MAX_EVENT_STREAM_BYTES = 8 * 1024 * 1024
EXPECTED_EVENT_KEYS = {
    "active",
    "duration_seconds",
    "gpu_millicelsius",
    "host_millicelsius",
    "observed_monotonic_seconds",
    "pause_index",
    "policy_sha256",
    "schema",
    "sequence",
    "started_monotonic_seconds",
    "transition",
}


class WslPacingEvidenceError(RuntimeError):
    """The WSL2 thermal-pacing evidence is absent, unsafe, or malformed."""


@dataclass(frozen=True)
class ThermalPause:
    pause_index: int
    started_monotonic_seconds: float
    completed_monotonic_seconds: float
    duration_seconds: float

    def overlap_seconds(self, started: float, finished: float) -> float:
        return max(
            0.0,
            min(self.completed_monotonic_seconds, finished)
            - max(self.started_monotonic_seconds, started),
        )


@dataclass(frozen=True)
class PacingSnapshot:
    records: tuple[dict[str, Any], ...]
    completed_pauses: tuple[ThermalPause, ...]

    def overlap_seconds(self, started: float, finished: float) -> float:
        if (
            not math.isfinite(started)
            or not math.isfinite(finished)
            or started < 0
            or finished < started
        ):
            raise WslPacingEvidenceError(
                "thermal-pacing overlap bounds are invalid"
            )
        return sum(
            pause.overlap_seconds(started, finished)
            for pause in self.completed_pauses
        )


def _read_stream(path: Path) -> bytes:
    if not path.is_absolute() or path != Path(os.path.normpath(path)):
        raise WslPacingEvidenceError(
            "WSL2 pacing event path must be normalized and absolute"
        )
    descriptor: int | None = None
    try:
        parent = path.parent.lstat()
        descriptor = os.open(
            path,
            os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW,
        )
        metadata = os.fstat(descriptor)
    except OSError as exc:
        raise WslPacingEvidenceError(
            f"cannot inspect WSL2 pacing event stream: {exc}"
        ) from exc
    try:
        if (
            not stat.S_ISREG(metadata.st_mode)
            or metadata.st_uid != os.getuid()
            or stat.S_IMODE(metadata.st_mode) != 0o400
            or not stat.S_ISDIR(parent.st_mode)
            or parent.st_uid != os.getuid()
            or stat.S_IMODE(parent.st_mode) & 0o077
        ):
            raise WslPacingEvidenceError(
                "WSL2 pacing event stream has unsafe type, ownership, or mode"
            )
        if metadata.st_size > MAX_EVENT_STREAM_BYTES:
            raise WslPacingEvidenceError(
                "WSL2 pacing event stream exceeds 8 MiB"
            )
        payload = bytearray()
        while chunk := os.read(descriptor, 65_536):
            payload.extend(chunk)
            if len(payload) > MAX_EVENT_STREAM_BYTES:
                raise WslPacingEvidenceError(
                    "WSL2 pacing event stream exceeds 8 MiB"
                )
        return bytes(payload)
    except OSError as exc:
        raise WslPacingEvidenceError(
            f"cannot read WSL2 pacing event stream: {exc}"
        ) from exc
    finally:
        if descriptor is not None:
            os.close(descriptor)


def _finite_number(value: Any) -> float | None:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
    ):
        return None
    return float(value)


def read_pacing_snapshot(
    source: Mapping[str, str],
    *,
    expected_policy_sha256: str,
) -> PacingSnapshot:
    raw_path = source.get(PACING_EVENTS_PATH_ENV)
    if not raw_path:
        raise WslPacingEvidenceError(
            "trusted WSL2 pacing event path is unavailable"
        )
    payload_bytes = _read_stream(Path(raw_path))
    try:
        payload = payload_bytes.decode("ascii")
    except UnicodeError as exc:
        raise WslPacingEvidenceError(
            f"WSL2 pacing event stream is not ASCII: {exc}"
        ) from exc
    if payload and not payload.endswith("\n"):
        raise WslPacingEvidenceError(
            "WSL2 pacing event stream ends with a partial record"
        )

    records: list[dict[str, Any]] = []
    completed_pauses: list[ThermalPause] = []
    active: tuple[int, float] | None = None
    next_pause_index = 1
    last_observed = -math.inf
    for sequence, line in enumerate(payload.splitlines()):
        try:
            value = strict_json_loads(line)
        except (UnicodeError, ValueError) as exc:
            raise WslPacingEvidenceError(
                f"WSL2 pacing event is malformed JSON: {exc}"
            ) from exc
        if not isinstance(value, dict) or set(value) != EXPECTED_EVENT_KEYS:
            raise WslPacingEvidenceError(
                "WSL2 pacing event violates its exact schema"
            )
        if (
            value["schema"] != PACING_EVENT_SCHEMA
            or value["policy_sha256"] != expected_policy_sha256
            or isinstance(value["sequence"], bool)
            or not isinstance(value["sequence"], int)
            or value["sequence"] != sequence
        ):
            raise WslPacingEvidenceError(
                "WSL2 pacing event schema, policy, or sequence drifted"
            )

        pause_index = value["pause_index"]
        started = _finite_number(value["started_monotonic_seconds"])
        observed = _finite_number(value["observed_monotonic_seconds"])
        duration = _finite_number(value["duration_seconds"])
        host = value["host_millicelsius"]
        gpu = value["gpu_millicelsius"]
        if (
            isinstance(pause_index, bool)
            or not isinstance(pause_index, int)
            or pause_index < 1
            or started is None
            or started < 0
            or observed is None
            or observed < started
            or observed < last_observed
            or duration is None
            or duration < 0
            or isinstance(host, bool)
            or not isinstance(host, int)
            or not -50_000 <= host <= 200_000
            or isinstance(gpu, bool)
            or not isinstance(gpu, int)
            or not 0 < gpu <= 200_000
        ):
            raise WslPacingEvidenceError(
                "WSL2 pacing event values are invalid"
            )

        transition = value["transition"]
        event_active = value["active"]
        if transition == "started":
            if (
                event_active is not True
                or duration != 0
                or active is not None
                or pause_index != next_pause_index
                or observed != started
            ):
                raise WslPacingEvidenceError(
                    "WSL2 pacing start transition is inconsistent"
                )
            active = (pause_index, started)
        elif transition == "completed":
            if event_active is not False or active != (pause_index, started):
                raise WslPacingEvidenceError(
                    "WSL2 pacing completion has no matching start"
                )
            expected_duration = observed - started
            if not math.isclose(
                duration,
                expected_duration,
                rel_tol=0.0,
                abs_tol=1e-6,
            ):
                raise WslPacingEvidenceError(
                    "WSL2 pacing duration does not match its interval"
                )
            completed_pauses.append(
                ThermalPause(
                    pause_index=pause_index,
                    started_monotonic_seconds=started,
                    completed_monotonic_seconds=observed,
                    duration_seconds=duration,
                )
            )
            active = None
            next_pause_index += 1
        else:
            raise WslPacingEvidenceError(
                f"unknown WSL2 pacing transition {transition!r}"
            )
        records.append(value)
        last_observed = observed

    if active is not None:
        raise WslPacingEvidenceError(
            "WSL2 pacing event stream ends during an active pause"
        )
    return PacingSnapshot(
        records=tuple(records),
        completed_pauses=tuple(completed_pauses),
    )
