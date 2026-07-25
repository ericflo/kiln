#!/usr/bin/env python3
"""Prepare a private Linux network namespace, then exec one qualification case."""

from __future__ import annotations

import ctypes
import os
import socket
import subprocess
import sys
from pathlib import Path


IP_CANDIDATES = (Path("/usr/sbin/ip"), Path("/sbin/ip"), Path("/bin/ip"))
LANDLOCK_CREATE_RULESET_VERSION = 1
LANDLOCK_RULE_PATH_BENEATH = 1
LANDLOCK_ACCESS_FS_EXECUTE = 1 << 0
LANDLOCK_ACCESS_FS_REFER = 1 << 13
LANDLOCK_MINIMUM_ABI = 2
PR_SET_NO_NEW_PRIVS = 38
SYS_LANDLOCK_CREATE_RULESET = 444
SYS_LANDLOCK_ADD_RULE = 445
SYS_LANDLOCK_RESTRICT_SELF = 446
EXECUTABLE_ROOT_CANDIDATES = (
    Path("/bin"),
    Path("/sbin"),
    Path("/usr"),
    Path("/home"),
    Path("/opt"),
    Path("/nix"),
    Path("/snap"),
    Path("/tmp"),
    Path("/var/tmp"),
    Path("/run/current-system"),
)


class LandlockRulesetAttr(ctypes.Structure):
    _fields_ = [("handled_access_fs", ctypes.c_uint64)]


class LandlockPathBeneathAttr(ctypes.Structure):
    _fields_ = [
        ("allowed_access", ctypes.c_uint64),
        ("parent_fd", ctypes.c_int32),
    ]


def fail(message: str) -> int:
    print(f"linux namespace setup failed: {message}", file=sys.stderr)
    return 125


def restrict_execution_to_native_roots() -> str | None:
    """Deny WSL's root-level /init interpreter while allowing native tools."""

    libc = ctypes.CDLL(None, use_errno=True)
    abi = libc.syscall(
        SYS_LANDLOCK_CREATE_RULESET,
        ctypes.c_void_p(),
        0,
        LANDLOCK_CREATE_RULESET_VERSION,
    )
    if abi < LANDLOCK_MINIMUM_ABI:
        return (
            f"Landlock ABI {LANDLOCK_MINIMUM_ABI}+ with REFER is unavailable "
            f"(reported {abi}, errno {ctypes.get_errno()})"
        )
    ruleset_attr = LandlockRulesetAttr(
        LANDLOCK_ACCESS_FS_EXECUTE | LANDLOCK_ACCESS_FS_REFER
    )
    ruleset_fd = libc.syscall(
        SYS_LANDLOCK_CREATE_RULESET,
        ctypes.byref(ruleset_attr),
        ctypes.sizeof(ruleset_attr),
        0,
    )
    if ruleset_fd < 0:
        return f"cannot create Landlock ruleset (errno {ctypes.get_errno()})"
    try:
        try:
            root_fd = os.open("/", os.O_PATH | os.O_CLOEXEC)
        except OSError as exc:
            return f"cannot open filesystem root for Landlock REFER: {exc}"
        try:
            root_rule = LandlockPathBeneathAttr(
                LANDLOCK_ACCESS_FS_REFER,
                root_fd,
            )
            if (
                libc.syscall(
                    SYS_LANDLOCK_ADD_RULE,
                    ruleset_fd,
                    LANDLOCK_RULE_PATH_BENEATH,
                    ctypes.byref(root_rule),
                    0,
                )
                != 0
            ):
                return (
                    "cannot allow filesystem-wide Landlock REFER "
                    f"(errno {ctypes.get_errno()})"
                )
        finally:
            os.close(root_fd)

        allowed = 0
        for path in EXECUTABLE_ROOT_CANDIDATES:
            try:
                path_fd = os.open(path, os.O_PATH | os.O_CLOEXEC)
            except OSError:
                continue
            try:
                rule = LandlockPathBeneathAttr(
                    LANDLOCK_ACCESS_FS_EXECUTE,
                    path_fd,
                )
                if (
                    libc.syscall(
                        SYS_LANDLOCK_ADD_RULE,
                        ruleset_fd,
                        LANDLOCK_RULE_PATH_BENEATH,
                        ctypes.byref(rule),
                        0,
                    )
                    != 0
                ):
                    return (
                        f"cannot allow native executable root {path} "
                        f"(errno {ctypes.get_errno()})"
                    )
                allowed += 1
            finally:
                os.close(path_fd)
        if allowed == 0:
            return "no native executable roots were available"
        if libc.prctl(PR_SET_NO_NEW_PRIVS, 1, 0, 0, 0) != 0:
            return f"cannot set no_new_privs (errno {ctypes.get_errno()})"
        if libc.syscall(SYS_LANDLOCK_RESTRICT_SELF, ruleset_fd, 0) != 0:
            return f"cannot apply Landlock ruleset (errno {ctypes.get_errno()})"
    finally:
        os.close(ruleset_fd)
    return None


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
    landlock_error = restrict_execution_to_native_roots()
    if landlock_error is not None:
        return fail(landlock_error)
    try:
        os.execvp(command[0], command)
    except OSError as exc:
        return fail(f"cannot execute {command[0]!r}: {exc}")


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
