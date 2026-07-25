from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path
from unittest import mock


QUALIFICATION_DIR = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "qualification_linux_namespace_exec",
    QUALIFICATION_DIR / "linux_namespace_exec.py",
)
assert SPEC is not None and SPEC.loader is not None
namespace_exec = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = namespace_exec
SPEC.loader.exec_module(namespace_exec)


class FakeLibc:
    def __init__(self, abi: int) -> None:
        self.abi = abi
        self.calls: list[tuple[object, ...]] = []

    def syscall(self, *args: object) -> int:
        self.calls.append(args)
        operation = args[0]
        if (
            operation == namespace_exec.SYS_LANDLOCK_CREATE_RULESET
            and args[-1] == namespace_exec.LANDLOCK_CREATE_RULESET_VERSION
        ):
            return self.abi
        if operation == namespace_exec.SYS_LANDLOCK_CREATE_RULESET:
            return 70
        return 0

    def prctl(self, *_args: object) -> int:
        return 0


class LinuxNamespaceExecTests(unittest.TestCase):
    def test_refer_is_allowed_globally_without_root_execute(self) -> None:
        libc = FakeLibc(namespace_exec.LANDLOCK_MINIMUM_ABI)
        opened: list[Path | str] = []

        def open_path(path: Path | str, _flags: int) -> int:
            opened.append(path)
            return 80 + len(opened)

        with mock.patch.object(
            namespace_exec.ctypes, "CDLL", return_value=libc
        ), mock.patch.object(
            namespace_exec.os, "open", side_effect=open_path
        ), mock.patch.object(
            namespace_exec.os, "close"
        ), mock.patch.object(
            namespace_exec,
            "EXECUTABLE_ROOT_CANDIDATES",
            (Path("/native"),),
        ):
            self.assertIsNone(namespace_exec.restrict_execution_to_native_roots())

        create = libc.calls[1]
        create_attr = create[1]._obj
        self.assertEqual(
            create_attr.handled_access_fs,
            namespace_exec.LANDLOCK_ACCESS_FS_EXECUTE
            | namespace_exec.LANDLOCK_ACCESS_FS_REFER,
        )
        rules = [
            call[3]._obj
            for call in libc.calls
            if call[0] == namespace_exec.SYS_LANDLOCK_ADD_RULE
        ]
        self.assertEqual(opened, ["/", Path("/native")])
        self.assertEqual(
            [rule.allowed_access for rule in rules],
            [
                namespace_exec.LANDLOCK_ACCESS_FS_REFER,
                namespace_exec.LANDLOCK_ACCESS_FS_EXECUTE,
            ],
        )

    def test_abi_without_refer_support_fails_closed(self) -> None:
        libc = FakeLibc(namespace_exec.LANDLOCK_MINIMUM_ABI - 1)
        with mock.patch.object(namespace_exec.ctypes, "CDLL", return_value=libc):
            error = namespace_exec.restrict_execution_to_native_roots()
        self.assertIsNotNone(error)
        assert error is not None
        self.assertIn("with REFER is unavailable", error)


if __name__ == "__main__":
    unittest.main()
