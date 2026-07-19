#!/usr/bin/env python3
"""Release one hash-bound executable through a supervisor-owned start gate."""

from __future__ import annotations

import argparse
import hashlib
import os
import stat
import sys
import time
from pathlib import Path
from typing import Any

from strict_json import loads as strict_json_loads


SCHEMA = "kiln.guarded-exec.v1"
GATE_PAYLOAD = b"go\n"
MAX_ARGUMENTS = 128
MAX_ARGUMENT_BYTES = 64 * 1024


class GuardedExecError(RuntimeError):
    """A guarded executable specification or start boundary is invalid."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return f"sha256:{digest.hexdigest()}"


def _sha256_descriptor(descriptor: int) -> str:
    digest = hashlib.sha256()
    os.lseek(descriptor, 0, os.SEEK_SET)
    while chunk := os.read(descriptor, 1024 * 1024):
        digest.update(chunk)
    os.lseek(descriptor, 0, os.SEEK_SET)
    return f"sha256:{digest.hexdigest()}"


def open_hash_bound_executable(path: Path, expected_sha256: str) -> int:
    try:
        descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
    except OSError as exc:
        raise GuardedExecError(f"cannot open guarded executable: {exc}") from exc
    try:
        metadata = os.fstat(descriptor)
        if not stat.S_ISREG(metadata.st_mode):
            raise GuardedExecError("guarded executable descriptor is not a regular file")
        if _sha256_descriptor(descriptor) != expected_sha256:
            raise GuardedExecError("guarded executable descriptor hash does not match the spec")
        return descriptor
    except BaseException:
        os.close(descriptor)
        raise


def _regular_absolute(path_text: Any, field: str, *, executable: bool = False) -> Path:
    if not isinstance(path_text, str) or not path_text or "\x00" in path_text:
        raise GuardedExecError(f"{field} must be a nonempty path string")
    path = Path(path_text)
    if not path.is_absolute() or path.is_symlink():
        raise GuardedExecError(f"{field} must be an absolute non-symlink path")
    try:
        metadata = path.stat()
    except OSError as exc:
        raise GuardedExecError(f"cannot inspect {field}: {exc}") from exc
    if not stat.S_ISREG(metadata.st_mode):
        raise GuardedExecError(f"{field} must be a regular file")
    if executable and not os.access(path, os.X_OK):
        raise GuardedExecError(f"{field} must be executable")
    return path


def load_spec(path: Path) -> tuple[Path, list[str], Path, dict[str, str], str]:
    if not path.is_absolute() or path.is_symlink() or not path.is_file():
        raise GuardedExecError("--spec must be an absolute non-symlink regular file")
    try:
        value = strict_json_loads(path.read_bytes())
    except Exception as exc:
        raise GuardedExecError(f"guarded exec spec is invalid JSON: {exc}") from exc
    fields = {"argv", "cwd", "environment", "executable", "schema"}
    if not isinstance(value, dict) or set(value) != fields or value["schema"] != SCHEMA:
        raise GuardedExecError("guarded exec spec fields or schema are invalid")
    executable_record = value["executable"]
    if not isinstance(executable_record, dict) or set(executable_record) != {"path", "sha256"}:
        raise GuardedExecError("guarded exec executable record is not closed")
    executable = _regular_absolute(
        executable_record["path"], "spec.executable.path", executable=True
    )
    expected_sha256 = executable_record["sha256"]
    if (
        not isinstance(expected_sha256, str)
        or len(expected_sha256) != 71
        or not expected_sha256.startswith("sha256:")
    ):
        raise GuardedExecError("spec.executable.sha256 is invalid")
    if _sha256(executable) != expected_sha256:
        raise GuardedExecError("guarded executable hash does not match the spec")
    argv = value["argv"]
    if (
        not isinstance(argv, list)
        or not argv
        or len(argv) > MAX_ARGUMENTS
        or any(not isinstance(item, str) or "\x00" in item for item in argv)
        or sum(len(item.encode("utf-8")) for item in argv) > MAX_ARGUMENT_BYTES
    ):
        raise GuardedExecError("spec.argv is invalid or exceeds its bound")
    if argv[0] != str(executable):
        raise GuardedExecError("spec.argv[0] must equal the executable path")
    cwd_text = value["cwd"]
    if not isinstance(cwd_text, str) or not cwd_text:
        raise GuardedExecError("spec.cwd must be a nonempty string")
    cwd = Path(cwd_text)
    if not cwd.is_absolute() or cwd.is_symlink() or not cwd.is_dir():
        raise GuardedExecError("spec.cwd must be an absolute non-symlink directory")
    environment = value["environment"]
    if not isinstance(environment, dict) or any(
        not isinstance(key, str)
        or not key
        or "=" in key
        or "\x00" in key
        or not isinstance(item, str)
        or "\x00" in item
        for key, item in environment.items()
    ):
        raise GuardedExecError("spec.environment must be a string map")
    return executable, argv, cwd, environment, expected_sha256


def wait_for_gate(path: Path, *, timeout_seconds: float = 60.0) -> None:
    if not path.is_absolute():
        raise GuardedExecError("--start-gate must be absolute")
    deadline = time.monotonic() + timeout_seconds
    while True:
        try:
            descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
        except FileNotFoundError:
            if time.monotonic() >= deadline:
                raise GuardedExecError("timed out waiting for the start gate")
            time.sleep(0.01)
            continue
        try:
            metadata = os.fstat(descriptor)
            if not stat.S_ISREG(metadata.st_mode) or metadata.st_nlink != 1:
                raise GuardedExecError("start gate must be a single-link regular file")
            if stat.S_IMODE(metadata.st_mode) & 0o077:
                raise GuardedExecError("start gate must not grant group or other permissions")
            payload = os.read(descriptor, len(GATE_PAYLOAD) + 1)
        finally:
            os.close(descriptor)
        if payload != GATE_PAYLOAD:
            raise GuardedExecError("start gate payload is invalid")
        return


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", required=True, type=Path)
    parser.add_argument("--start-gate", required=True, type=Path)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    descriptor: int | None = None
    try:
        executable, command, cwd, environment, expected_sha256 = load_spec(args.spec)
        descriptor = open_hash_bound_executable(executable, expected_sha256)
        wait_for_gate(args.start_gate)
        if _sha256_descriptor(descriptor) != expected_sha256:
            raise GuardedExecError("guarded executable changed while waiting for release")
        os.chdir(cwd)
        os.execve(f"/proc/self/fd/{descriptor}", command, environment)
    except BaseException as exc:
        print(f"guarded exec failed: {exc}", file=sys.stderr)
        return 1
    finally:
        if descriptor is not None:
            os.close(descriptor)
    raise AssertionError("os.execve returned")


if __name__ == "__main__":
    raise SystemExit(main())
