#!/usr/bin/env python3
"""Keep a systemd scope alive until its outer controller seals accounting."""

from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Sequence


class PayloadError(RuntimeError):
    """The trusted scope payload could not preserve its handoff contract."""


def _publish(path: Path, value: dict[str, Any]) -> None:
    payload = (
        json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        )
        + "\n"
    ).encode("ascii")
    temporary: Path | None = None
    try:
        descriptor, raw_path = tempfile.mkstemp(
            prefix=f".{path.name}.",
            suffix=".tmp",
            dir=path.parent,
        )
        temporary = Path(raw_path)
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            os.fchmod(handle.fileno(), 0o600)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary, path, follow_symlinks=False)
        except FileExistsError as exc:
            raise PayloadError(f"refusing to replace status file: {path}") from exc
    except OSError as exc:
        raise PayloadError(f"cannot publish scope status {path}: {exc}") from exc
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def execute(status_path: Path, release_path: Path, command: list[str]) -> int:
    try:
        parent = status_path.parent.resolve(strict=True)
        if release_path.parent.resolve(strict=True) != parent:
            raise PayloadError("status and release paths must share one directory")
        metadata = parent.stat()
    except OSError as exc:
        raise PayloadError(f"cannot resolve payload directory: {exc}") from exc
    if metadata.st_uid != os.getuid() or metadata.st_mode & 0o077:
        raise PayloadError("payload directory must be private and owned")
    if status_path.exists() or release_path.exists():
        raise PayloadError("payload handoff paths must not already exist")
    try:
        process = subprocess.Popen(command)
    except OSError as exc:
        raise PayloadError(f"cannot launch contained scope command: {exc}") from exc

    interrupted: int | None = None

    def handle_signal(signum: int, _frame: Any) -> None:
        nonlocal interrupted
        interrupted = signum
        if process.poll() is None:
            process.terminate()

    old_handlers = {
        signum: signal.signal(signum, handle_signal)
        for signum in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP)
    }
    try:
        returncode = process.wait()
        _publish(
            status_path,
            {
                "schema": "kiln.wsl2-scope-payload-status.v1",
                "returncode": returncode,
            },
        )
        while not release_path.exists():
            if interrupted is not None:
                return 128 + interrupted
            time.sleep(0.005)
    finally:
        for signum, handler in old_handlers.items():
            signal.signal(signum, handler)
        if process.poll() is None:
            process.kill()
            process.wait()
    return returncode if returncode >= 0 else 128 - returncode


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--status-path", type=Path, required=True)
    parser.add_argument("--release-path", type=Path, required=True)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args(argv)
    if args.command and args.command[0] == "--":
        args.command = args.command[1:]
    if not args.command:
        parser.error("a command is required after --")
    if not args.status_path.is_absolute() or not args.release_path.is_absolute():
        parser.error("handoff paths must be absolute")
    return args


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        return execute(args.status_path, args.release_path, args.command)
    except PayloadError as exc:
        print(f"error: WSL2 scope payload: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
