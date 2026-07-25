#!/usr/bin/env python3
"""Prepare a private Linux network namespace, then exec one qualification case."""

from __future__ import annotations

import os
import socket
import subprocess
import sys
from pathlib import Path


IP_CANDIDATES = (Path("/usr/sbin/ip"), Path("/sbin/ip"), Path("/bin/ip"))


def fail(message: str) -> int:
    print(f"linux namespace setup failed: {message}", file=sys.stderr)
    return 125


def main(argv: list[str]) -> int:
    if len(argv) < 2 or argv[0] != "--":
        return fail("expected -- followed by a command")
    command = argv[1:]
    ip = next((path for path in IP_CANDIDATES if path.is_file()), None)
    if ip is None:
        return fail("the ip utility is required to enable private loopback")
    try:
        completed = subprocess.run(
            [str(ip), "link", "set", "dev", "lo", "up"],
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
            timeout=5,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        return fail(f"cannot enable private loopback: {exc}")
    if completed.returncode != 0:
        detail = completed.stderr.decode(errors="replace").strip()
        return fail(
            f"cannot enable private loopback (exit {completed.returncode})"
            + (f": {detail}" if detail else "")
        )
    interfaces = [name for _index, name in socket.if_nameindex()]
    if interfaces != ["lo"]:
        return fail(f"private namespace interfaces are not exactly ['lo']: {interfaces!r}")
    try:
        os.execvp(command[0], command)
    except OSError as exc:
        return fail(f"cannot execute {command[0]!r}: {exc}")


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
