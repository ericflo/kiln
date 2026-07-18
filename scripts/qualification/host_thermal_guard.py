"""Fail-closed host thermal pacing for local hardware qualification."""

from __future__ import annotations

import dataclasses
import os
import re
import signal
import subprocess
import threading
import time
from pathlib import Path
from typing import Any, Callable


class ThermalGuardError(RuntimeError):
    """Raised when a thermal guard policy is internally invalid."""


TraceCallback = Callable[..., None]


@dataclasses.dataclass(frozen=True)
class ThermalPacingEvent:
    observed: float
    category: str
    message: str
    fields: dict[str, Any] = dataclasses.field(default_factory=dict)


@dataclasses.dataclass
class ThermalPacingEvidence:
    error_type: type[Exception] = ThermalGuardError
    event_count: int = 0
    completed_event_count: int = 0
    total_seconds: float = 0.0
    max_seconds: float = 0.0
    max_start_millicelsius: int = 0
    active: bool = False
    _lock: threading.Lock = dataclasses.field(
        default_factory=threading.Lock,
        init=False,
        repr=False,
    )

    def begin(self, temperature_millicelsius: int) -> None:
        with self._lock:
            if self.active:
                raise self.error_type("thermal pacing began while already active")
            self.event_count += 1
            self.max_start_millicelsius = max(
                self.max_start_millicelsius, temperature_millicelsius
            )
            self.active = True

    def finish(self, elapsed_seconds: float, *, completed: bool) -> None:
        with self._lock:
            if not self.active:
                raise self.error_type("thermal pacing finished while inactive")
            if completed:
                self.completed_event_count += 1
            self.total_seconds += elapsed_seconds
            self.max_seconds = max(self.max_seconds, elapsed_seconds)
            self.active = False

    def metric_values(self) -> dict[str, float | int]:
        with self._lock:
            return {
                "host_thermal_pacing_active_end": int(self.active),
                "host_thermal_pacing_completed_event_count": (
                    self.completed_event_count
                ),
                "host_thermal_pacing_event_count": self.event_count,
                "host_thermal_pacing_max_seconds": self.max_seconds,
                "host_thermal_pacing_max_start_millicelsius": (
                    self.max_start_millicelsius
                ),
                "host_thermal_pacing_seconds": self.total_seconds,
            }


def resolve_hwmon_temperature_input(
    hwmon_name: str,
    label: str,
    hwmon_root: Path = Path("/sys/class/hwmon"),
    *,
    error_type: type[Exception] = ThermalGuardError,
) -> Path:
    if not hwmon_name or not label:
        raise error_type("hwmon temperature selector requires nonempty name and label")
    matches: list[Path] = []
    for device in sorted(hwmon_root.glob("hwmon*")):
        name_path = device / "name"
        if not name_path.is_file():
            continue
        if name_path.read_text(encoding="utf-8").strip() != hwmon_name:
            continue
        for label_path in sorted(device.glob("temp*_label")):
            if label_path.read_text(encoding="utf-8").strip() != label:
                continue
            input_path = label_path.with_name(
                label_path.name.removesuffix("_label") + "_input"
            )
            if not input_path.is_file():
                raise error_type(
                    f"hwmon sensor {hwmon_name}/{label} has no temperature input"
                )
            matches.append(input_path)
    if len(matches) != 1:
        raise error_type(
            f"hwmon sensor {hwmon_name}/{label} resolved to {len(matches)} inputs, "
            "expected exactly one"
        )
    return matches[0]


def read_hwmon_temperature_millicelsius(
    path: Path,
    *,
    error_type: type[Exception] = ThermalGuardError,
) -> int:
    raw = path.read_text(encoding="utf-8").strip()
    if re.fullmatch(r"[+-]?\d+", raw) is None:
        raise error_type(f"{path} contains a non-integer temperature {raw!r}")
    value = int(raw)
    if value < -100_000 or value > 250_000:
        raise error_type(f"{path} temperature {value} millicelsius is implausible")
    return value


class HostThermalGuard:
    """Continuously pace one process group below a hard host temperature limit."""

    def __init__(
        self,
        process: subprocess.Popen[Any],
        *,
        hwmon_name: str,
        label: str,
        limit_millicelsius: int,
        pacing_start_millicelsius: int | None = None,
        pacing_resume_millicelsius: int | None = None,
        poll_interval_seconds: float = 0.25,
        hwmon_root: Path = Path("/sys/class/hwmon"),
        trace_callback: TraceCallback | None = None,
        error_type: type[Exception] = ThermalGuardError,
    ) -> None:
        if limit_millicelsius <= 0:
            raise error_type("host thermal limit must be positive")
        if poll_interval_seconds <= 0:
            raise error_type("host thermal poll interval must be positive")
        if (pacing_start_millicelsius is None) != (
            pacing_resume_millicelsius is None
        ):
            raise error_type("host thermal pacing requires both start and resume")
        if (
            pacing_start_millicelsius is not None
            and pacing_resume_millicelsius is not None
            and not (
                0
                < pacing_resume_millicelsius
                < pacing_start_millicelsius
                < limit_millicelsius
            )
        ):
            raise error_type(
                "host thermal pacing requires 0 < resume < start < safety limit"
            )
        self.process = process
        self.hwmon_name = hwmon_name
        self.label = label
        self.limit_millicelsius = limit_millicelsius
        self.pacing_start_millicelsius = pacing_start_millicelsius
        self.pacing_resume_millicelsius = pacing_resume_millicelsius
        self.poll_interval_seconds = poll_interval_seconds
        self.pacing_evidence = ThermalPacingEvidence(error_type=error_type)
        self.hwmon_root = hwmon_root
        self.trace_callback = trace_callback
        self.error_type = error_type
        self.input_path: Path | None = None
        self.stop = threading.Event()
        self.samples: list[int] = []
        self.errors: list[str] = []
        self.trip_reason: str | None = None
        self.thread = threading.Thread(
            target=self._run, name="qualification-host-thermal-guard"
        )
        self._started = False
        self._closed = False
        self._pacing_started_at: float | None = None
        self._pacing_start_temperature = 0
        self._pacing_phase = "startup"
        self._pacing_events: list[ThermalPacingEvent] = []
        self._pacing_lock = threading.Lock()

    def _trace(self, event: str, **fields: Any) -> None:
        if self.trace_callback is not None:
            self.trace_callback(event, **fields)

    def start(self) -> None:
        self._sample()
        if self.trip_reason is not None:
            return
        assert self.input_path is not None
        self._trace(
            "host_thermal_guard_armed",
            sensor_name=self.hwmon_name,
            sensor_label=self.label,
            sensor_path=str(self.input_path),
            limit_millicelsius=self.limit_millicelsius,
            poll_interval_ms=int(self.poll_interval_seconds * 1000),
        )
        if self.pacing_start_millicelsius is not None:
            self._trace(
                "host_thermal_pacing_armed",
                mode="continuous_process_group_stop",
                start_millicelsius=self.pacing_start_millicelsius,
                resume_millicelsius=self.pacing_resume_millicelsius,
                pause_signal="SIGSTOP",
                resume_signal="SIGCONT",
            )
        self.thread.start()
        self._started = True

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self.stop.set()
        if self._started:
            self.thread.join(timeout=10.0)
            if self.thread.is_alive():
                message = "host-thermal-guard thread did not stop within 10 seconds"
                if len(self.errors) < 8:
                    self.errors.append(message)
                self._trip(message)
        if self._pacing_started_at is not None:
            self._resume_pacing(interrupted=True)
        if self.trip_reason is None:
            self._sample(allow_pacing=False)

    def metric_values(self) -> dict[str, int]:
        if not self.samples:
            return {
                "host_temperature_end_millicelsius": 0,
                "host_temperature_peak_millicelsius": 0,
                "host_temperature_start_millicelsius": 0,
                "host_thermal_guard_trip_count": int(self.trip_reason is not None),
            }
        return {
            "host_temperature_end_millicelsius": self.samples[-1],
            "host_temperature_peak_millicelsius": max(self.samples),
            "host_temperature_start_millicelsius": self.samples[0],
            "host_thermal_guard_trip_count": int(self.trip_reason is not None),
        }

    def pacing_metric_values(self) -> dict[str, float | int]:
        return self.pacing_evidence.metric_values()

    def set_phase(self, phase: str) -> None:
        if not phase:
            raise self.error_type("host thermal pacing phase must be nonempty")
        with self._pacing_lock:
            self._pacing_phase = phase

    def pacing_events_since(self, started: float) -> list[ThermalPacingEvent]:
        with self._pacing_lock:
            return [event for event in self._pacing_events if event.observed >= started]

    def _trip(self, reason: str) -> None:
        if self.trip_reason is not None:
            return
        self.trip_reason = reason
        pacing_active = self._pacing_started_at is not None
        if self.process.poll() is None:
            try:
                os.killpg(self.process.pid, signal.SIGTERM)
            except OSError as exc:
                if not isinstance(exc, ProcessLookupError) and len(self.errors) < 8:
                    self.errors.append(f"failed to terminate server process group: {exc}")
            if pacing_active:
                try:
                    os.killpg(self.process.pid, signal.SIGCONT)
                except OSError as exc:
                    if not isinstance(exc, ProcessLookupError) and len(self.errors) < 8:
                        self.errors.append(
                            f"failed to release paced server process group: {exc}"
                        )
        if pacing_active:
            self._record_pacing_finish(interrupted=True)

    def _pause_pacing(self, temperature: int) -> None:
        if self._pacing_started_at is not None or self.process.poll() is not None:
            return
        try:
            os.killpg(self.process.pid, signal.SIGSTOP)
        except ProcessLookupError:
            return
        except OSError as exc:
            message = f"failed to pause server process group: {exc}"
            if len(self.errors) < 8:
                self.errors.append(message)
            self._trip(message)
            return
        started = time.monotonic()
        with self._pacing_lock:
            phase = self._pacing_phase
            self._pacing_started_at = started
            self._pacing_start_temperature = temperature
            self._pacing_events.append(
                ThermalPacingEvent(
                    started,
                    "host_thermal_pacing",
                    "host thermal pacing paused the server process group",
                    {"phase": phase, "temperature_millicelsius": temperature},
                )
            )
        self.pacing_evidence.begin(temperature)
        self._trace(
            "host_thermal_pacing_started",
            phase=phase,
            temperature_millicelsius=temperature,
            start_millicelsius=self.pacing_start_millicelsius,
            resume_millicelsius=self.pacing_resume_millicelsius,
            scope="server_process_group",
        )

    def _record_pacing_finish(self, *, interrupted: bool) -> None:
        with self._pacing_lock:
            started = self._pacing_started_at
            if started is None:
                return
            finished = time.monotonic()
            elapsed = finished - started
            phase = self._pacing_phase
            start_temperature = self._pacing_start_temperature
            self._pacing_started_at = None
            self._pacing_start_temperature = 0
            self._pacing_events.append(
                ThermalPacingEvent(
                    finished,
                    "host_thermal_pacing",
                    "host thermal pacing resumed the server process group",
                    {
                        "duration_seconds": elapsed,
                        "interrupted": interrupted,
                        "phase": phase,
                    },
                )
            )
        self.pacing_evidence.finish(elapsed, completed=not interrupted)
        self._trace(
            (
                "host_thermal_pacing_interrupted"
                if interrupted
                else "host_thermal_pacing_completed"
            ),
            phase=phase,
            duration_seconds=elapsed,
            start_temperature_millicelsius=start_temperature,
        )

    def _resume_pacing(self, *, interrupted: bool = False) -> None:
        if self._pacing_started_at is None:
            return
        if self.process.poll() is None:
            try:
                os.killpg(self.process.pid, signal.SIGCONT)
            except ProcessLookupError:
                pass
            except OSError as exc:
                message = f"failed to resume server process group: {exc}"
                if len(self.errors) < 8:
                    self.errors.append(message)
                self._trip(message)
                return
        self._record_pacing_finish(interrupted=interrupted)

    def _sample(self, *, allow_pacing: bool = True) -> None:
        try:
            if self.input_path is None:
                self.input_path = resolve_hwmon_temperature_input(
                    self.hwmon_name,
                    self.label,
                    self.hwmon_root,
                    error_type=self.error_type,
                )
            temperature = read_hwmon_temperature_millicelsius(
                self.input_path, error_type=self.error_type
            )
            self.samples.append(temperature)
            if temperature >= self.limit_millicelsius:
                self._trip(
                    f"host {self.hwmon_name}/{self.label} reached {temperature} "
                    f"millicelsius at or above the {self.limit_millicelsius}-millicelsius "
                    "safety limit"
                )
                return
            if not allow_pacing or self.pacing_start_millicelsius is None:
                return
            assert self.pacing_resume_millicelsius is not None
            if self._pacing_started_at is None:
                if temperature >= self.pacing_start_millicelsius:
                    self._pause_pacing(temperature)
            elif temperature <= self.pacing_resume_millicelsius:
                self._resume_pacing()
        except Exception as exc:
            message = f"{type(exc).__name__}: {exc}"
            if len(self.errors) < 8:
                self.errors.append(message)
            self._trip(f"host thermal guard failed closed: {message}")

    def _run(self) -> None:
        while not self.stop.wait(self.poll_interval_seconds):
            self._sample()
            if self.trip_reason is not None:
                return
