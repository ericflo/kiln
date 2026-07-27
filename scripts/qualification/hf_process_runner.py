#!/usr/bin/env python3
"""Run one gated Hugging Face worker with bounded process cleanup."""

from __future__ import annotations

import argparse
import json
import math
import os
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

from strict_json import loads as strict_json_loads


SCHEMA = "kiln.hf-process-containment.v1"
PASS_PREFIX = "KILN_HF_PROCESS_CONTAINMENT_PASS "
WORKER_TIMEOUT_SECONDS = 570.0


class RunnerError(RuntimeError):
    """The worker could not be run inside the declared process boundary."""


def _validate_worker_command(command: list[str]) -> list[str]:
    if command and command[0] == "--":
        command = command[1:]
    if len(command) < 2:
        raise RunnerError("worker command must name an absolute Python and script path")
    if any("\x00" in item for item in command):
        raise RunnerError("worker command contains a NUL byte")
    python = Path(command[0])
    script = Path(command[1])
    if not python.is_absolute() or not script.is_absolute():
        raise RunnerError("worker Python and script paths must be absolute")
    if not python.is_file() or not os.access(python, os.X_OK):
        raise RunnerError("worker Python must resolve to an executable file")
    if script.is_symlink() or not script.is_file():
        raise RunnerError("worker script must be a non-symlink regular file")
    if "--start-gate" in command:
        raise RunnerError("worker command must not provide its own --start-gate")
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


def run_contained(
    *,
    workspace: Path,
    worker_command: list[str],
    worker_environment: dict[str, str] | None = None,
    timeout_seconds: float = WORKER_TIMEOUT_SECONDS,
) -> tuple[int, str, str, dict[str, Any]]:
    if not workspace.is_absolute():
        raise RunnerError("workspace path must be absolute")
    if workspace.is_symlink() or not workspace.is_dir():
        raise RunnerError("workspace must be a non-symlink directory")
    if not math.isfinite(timeout_seconds) or timeout_seconds <= 0:
        raise RunnerError("worker timeout must be finite and positive")
    command = _validate_worker_command(worker_command)
    gate = workspace / "hf-worker.start-gate"
    if gate.exists() or gate.is_symlink():
        raise RunnerError(f"refusing stale worker start gate {gate}")
    started = time.monotonic()
    process = subprocess.Popen(
        [*command, "--start-gate", str(gate)],
        cwd=Path.cwd(),
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        start_new_session=True,
        env=worker_environment,
    )
    try:
        _release_gate(gate)
        stdout, stderr = process.communicate(timeout=timeout_seconds)
    except BaseException:
        _terminate_group(process)
        raise
    finally:
        if process.poll() is None:
            _terminate_group(process)
        gate.unlink(missing_ok=True)
    evidence = {
        "elapsed_seconds": time.monotonic() - started,
        "schema": SCHEMA,
        "timeout_seconds": timeout_seconds,
        "worker_exit_code": process.returncode,
    }
    return process.returncode, stdout, stderr, evidence


def validate_evidence(evidence: Any) -> dict[str, Any]:
    if not isinstance(evidence, dict) or set(evidence) != {
        "elapsed_seconds",
        "schema",
        "timeout_seconds",
        "worker_exit_code",
    }:
        raise RunnerError("HF process-containment marker fields are not closed")
    if evidence["schema"] != SCHEMA or evidence["worker_exit_code"] != 0:
        raise RunnerError("HF process-containment marker does not report a clean worker")
    for name in ("elapsed_seconds", "timeout_seconds"):
        value = evidence[name]
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
            or value < 0
        ):
            raise RunnerError(f"HF process-containment {name} is invalid")
    if evidence["timeout_seconds"] <= 0:
        raise RunnerError("HF process-containment timeout must be positive")
    return evidence


def parse_pass_marker(output: str) -> dict[str, Any]:
    records = [
        line[len(PASS_PREFIX) :]
        for line in output.splitlines()
        if line.startswith(PASS_PREFIX)
    ]
    if len(records) != 1:
        raise RunnerError(
            f"expected one HF process-containment marker, found {len(records)}"
        )
    try:
        evidence = strict_json_loads(records[0])
    except Exception as exc:
        raise RunnerError(
            f"HF process-containment marker is invalid JSON: {exc}"
        ) from exc
    return validate_evidence(evidence)


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", required=True, type=Path)
    parser.add_argument(
        "--timeout-seconds", type=float, default=WORKER_TIMEOUT_SECONDS
    )
    parser.add_argument("worker_command", nargs=argparse.REMAINDER)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    try:
        returncode, stdout, stderr, evidence = run_contained(
            workspace=args.workspace,
            worker_command=args.worker_command,
            timeout_seconds=args.timeout_seconds,
        )
    except BaseException as exc:
        print(f"HF process runner failed: {exc}", file=sys.stderr)
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
