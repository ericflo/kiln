#!/usr/bin/env python3
"""Supervise one gated HF worker with continuous host thermal containment."""

from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import host_thermal_guard as thermal
import host_thermal_policy as thermal_policy_file
from strict_json import loads as strict_json_loads


SCHEMA = "kiln.hf-thermal-containment.v1"
PASS_PREFIX = "KILN_HF_THERMAL_CONTAINMENT_PASS "
WORKER_TIMEOUT_SECONDS = 570


class SupervisorError(RuntimeError):
    """The worker could not be run inside the declared safety boundary."""


def _trace(event: str, **fields: Any) -> None:
    print(
        json.dumps(
            {"event": event, **fields},
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        ),
        file=sys.stderr,
        flush=True,
    )


def _validate_worker_command(command: list[str]) -> list[str]:
    if command and command[0] == "--":
        command = command[1:]
    if len(command) < 2:
        raise SupervisorError("worker command must name an absolute Python and script path")
    if any("\x00" in item for item in command):
        raise SupervisorError("worker command contains a NUL byte")
    python = Path(command[0])
    script = Path(command[1])
    if not python.is_absolute() or not script.is_absolute():
        raise SupervisorError("worker Python and script paths must be absolute")
    if not python.is_file() or not os.access(python, os.X_OK):
        raise SupervisorError("worker Python must resolve to an executable file")
    if script.is_symlink() or not script.is_file():
        raise SupervisorError("worker script must be a non-symlink regular file")
    if "--start-gate" in command:
        raise SupervisorError("worker command must not provide its own --start-gate")
    return command


def _terminate_group(process: subprocess.Popen[str]) -> None:
    if process.poll() is not None:
        return
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        pass
    try:
        process.wait(timeout=15)
        return
    except subprocess.TimeoutExpired:
        pass
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    process.wait(timeout=15)


def _release_gate(path: Path) -> None:
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        with os.fdopen(descriptor, "wb", closefd=False) as handle:
            handle.write(b"go\n")
            handle.flush()
            os.fsync(handle.fileno())
    finally:
        os.close(descriptor)


def supervise(
    *,
    policy_path: Path,
    workspace: Path,
    worker_command: list[str],
) -> tuple[int, str, str, dict[str, Any]]:
    if not policy_path.is_absolute() or not workspace.is_absolute():
        raise SupervisorError("policy and workspace paths must be absolute")
    if workspace.is_symlink() or not workspace.is_dir():
        raise SupervisorError("workspace must be a non-symlink directory")
    policy_record, policy, settlement_timeout = thermal_policy_file.load(
        policy_path,
        error_type=SupervisorError,
        cooldown_mode="post_process_exit_consecutive_samples",
    )
    command = _validate_worker_command(worker_command)
    prelaunch = thermal_policy_file.wait_for_prelaunch_cooldown(
        policy,
        trace_callback=_trace,
        error_type=SupervisorError,
    )
    gate = workspace / "hf-worker.start-gate"
    if gate.exists() or gate.is_symlink():
        raise SupervisorError(f"refusing stale worker start gate {gate}")
    process = subprocess.Popen(
        [*command, "--start-gate", str(gate)],
        cwd=Path.cwd(),
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        start_new_session=True,
    )
    guard: thermal.HostThermalGuard | None = None
    stdout = ""
    stderr = ""
    try:
        guard = thermal.HostThermalGuard(
            process,
            **policy.guard_kwargs(),
            trace_callback=_trace,
            error_type=SupervisorError,
        )
        guard.set_phase("startup-gate")
        guard.start()
        if guard.trip_reason is not None:
            raise SupervisorError(guard.trip_reason)
        _release_gate(gate)
        guard.set_phase("hf-eager-forward")
        stdout, stderr = process.communicate(timeout=WORKER_TIMEOUT_SECONDS)
    except BaseException:
        _terminate_group(process)
        raise
    finally:
        if process.poll() is None:
            _terminate_group(process)
        if guard is not None:
            guard.close()
        gate.unlink(missing_ok=True)
    assert guard is not None
    if guard.trip_reason is not None:
        raise SupervisorError(f"host thermal guard tripped: {guard.trip_reason}")
    if guard.errors:
        raise SupervisorError("host thermal guard errors: " + " | ".join(guard.errors))
    metrics = {**guard.metric_values(), **guard.pacing_metric_values()}
    if metrics["host_thermal_guard_trip_count"] != 0:
        raise SupervisorError("host thermal guard reported a trip")
    if metrics["host_thermal_cooldown_completed_count"] != 1:
        raise SupervisorError("host thermal cooldown did not complete exactly once")
    if metrics["host_thermal_cooldown_timeout_count"] != 0:
        raise SupervisorError("host thermal cooldown timed out")
    if metrics["host_thermal_pacing_active_end"] != 0:
        raise SupervisorError("host thermal pacing remained active after worker exit")
    if (
        metrics["host_thermal_pacing_event_count"]
        != metrics["host_thermal_pacing_completed_event_count"]
    ):
        raise SupervisorError("host thermal pacing intervals did not reconcile")
    evidence = {
        "phase_settlement_timeout_seconds": settlement_timeout,
        "policy": policy_record,
        "prelaunch_cooldown": prelaunch,
        "runtime": metrics,
        "schema": SCHEMA,
        "worker_exit_code": process.returncode,
    }
    return process.returncode, stdout, stderr, evidence


def validate_evidence(evidence: Any) -> dict[str, Any]:
    if not isinstance(evidence, dict) or set(evidence) != {
        "phase_settlement_timeout_seconds",
        "policy",
        "prelaunch_cooldown",
        "runtime",
        "schema",
        "worker_exit_code",
    }:
        raise SupervisorError("HF thermal-containment marker fields are not closed")
    if evidence["schema"] != SCHEMA or evidence["worker_exit_code"] != 0:
        raise SupervisorError("HF thermal-containment marker does not report a clean worker")
    policy_record, policy, settlement = thermal_policy_file.validate(
        evidence["policy"],
        "HF thermal-containment policy",
        error_type=SupervisorError,
        cooldown_mode="post_process_exit_consecutive_samples",
    )
    if (
        policy_record != evidence["policy"]
        or settlement != evidence["phase_settlement_timeout_seconds"]
    ):
        raise SupervisorError("HF thermal-containment policy evidence is inconsistent")
    prelaunch = evidence["prelaunch_cooldown"]
    prelaunch_fields = {
        "completed",
        "elapsed_seconds",
        "poll_interval_ms",
        "sample_count",
        "scope",
        "sensor_path",
        "stable_samples_observed",
        "stable_samples_required",
        "target_millicelsius",
        "temperature_end_millicelsius",
        "temperature_peak_millicelsius",
        "temperature_start_millicelsius",
        "timeout_seconds",
    }
    if not isinstance(prelaunch, dict) or set(prelaunch) != prelaunch_fields:
        raise SupervisorError("HF thermal-containment prelaunch fields are not closed")
    if (
        prelaunch["completed"] is not True
        or prelaunch["scope"] != "host_package_before_process_creation"
        or prelaunch["poll_interval_ms"] != policy.poll_interval_ms
        or prelaunch["target_millicelsius"] != policy.cooldown_target_millicelsius
        or prelaunch["stable_samples_required"] != policy.cooldown_stable_samples
        or prelaunch["stable_samples_observed"] < policy.cooldown_stable_samples
        or prelaunch["temperature_end_millicelsius"] > policy.cooldown_target_millicelsius
    ):
        raise SupervisorError("HF thermal-containment prelaunch evidence is inconsistent")
    runtime = evidence["runtime"]
    runtime_fields = {
        "host_temperature_end_millicelsius",
        "host_temperature_peak_millicelsius",
        "host_temperature_start_millicelsius",
        "host_thermal_cooldown_active_end",
        "host_thermal_cooldown_completed_count",
        "host_thermal_cooldown_peak_millicelsius",
        "host_thermal_cooldown_sample_count",
        "host_thermal_cooldown_seconds",
        "host_thermal_cooldown_stable_sample_count",
        "host_thermal_cooldown_timeout_count",
        "host_thermal_guard_trip_count",
        "host_thermal_pacing_active_end",
        "host_thermal_pacing_completed_event_count",
        "host_thermal_pacing_event_count",
        "host_thermal_pacing_max_seconds",
        "host_thermal_pacing_max_start_millicelsius",
        "host_thermal_pacing_seconds",
    }
    if not isinstance(runtime, dict) or set(runtime) != runtime_fields:
        raise SupervisorError("HF thermal-containment runtime fields are not closed")
    if runtime.get("host_thermal_guard_trip_count") != 0:
        raise SupervisorError("HF thermal-containment marker reports a guard trip")
    if runtime.get("host_thermal_cooldown_completed_count") != 1:
        raise SupervisorError("HF thermal-containment marker reports incomplete cooldown")
    if (
        runtime["host_thermal_cooldown_active_end"] != 0
        or runtime["host_thermal_cooldown_timeout_count"] != 0
        or runtime["host_thermal_pacing_active_end"] != 0
        or runtime["host_thermal_pacing_event_count"]
        != runtime["host_thermal_pacing_completed_event_count"]
        or runtime["host_temperature_peak_millicelsius"] >= policy.limit_millicelsius
    ):
        raise SupervisorError("HF thermal-containment runtime did not close safely")
    return evidence


def parse_pass_marker(output: str) -> dict[str, Any]:
    records = [
        line[len(PASS_PREFIX) :]
        for line in output.splitlines()
        if line.startswith(PASS_PREFIX)
    ]
    if len(records) != 1:
        raise SupervisorError(
            f"expected one HF thermal-containment marker, found {len(records)}"
        )
    try:
        evidence = strict_json_loads(records[0])
    except Exception as exc:
        raise SupervisorError(f"HF thermal-containment marker is invalid JSON: {exc}") from exc
    return validate_evidence(evidence)


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host-thermal-policy", required=True, type=Path)
    parser.add_argument("--workspace", required=True, type=Path)
    parser.add_argument("worker_command", nargs=argparse.REMAINDER)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    try:
        returncode, stdout, stderr, evidence = supervise(
            policy_path=args.host_thermal_policy,
            workspace=args.workspace,
            worker_command=args.worker_command,
        )
    except BaseException as exc:
        print(f"HF thermal supervisor failed: {exc}", file=sys.stderr)
        return 1
    sys.stdout.write(stdout)
    sys.stderr.write(stderr)
    if returncode != 0:
        print(f"HF worker exited {returncode}", file=sys.stderr)
        return returncode if 0 < returncode <= 125 else 1
    print(
        PASS_PREFIX
        + json.dumps(evidence, allow_nan=False, separators=(",", ":"), sort_keys=True),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
