from __future__ import annotations

import importlib.util
import contextlib
import io
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock


QUALIFICATION_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(QUALIFICATION_DIR))
SPEC = importlib.util.spec_from_file_location(
    "qualification_wsl_scope_exec",
    QUALIFICATION_DIR / "wsl_scope_exec.py",
)
assert SPEC is not None and SPEC.loader is not None
scope = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = scope
SPEC.loader.exec_module(scope)


class WslScopeExecTests(unittest.TestCase):
    def test_runtime_bus_units_are_no_clobber_and_exact(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "bus.socket"
            scope._atomic_runtime_unit(path, "exact\n")
            self.assertEqual(path.read_text(encoding="ascii"), "exact\n")
            scope._atomic_runtime_unit(path, "exact\n")
            with self.assertRaisesRegex(scope.ScopeExecError, "refusing to replace"):
                scope._atomic_runtime_unit(path, "different\n")

    def test_namespace_command_contract_is_closed(self) -> None:
        command = [
            "/usr/bin/unshare",
            "--user",
            "--map-root-user",
            "--net",
            "--pid",
            "--fork",
            "--kill-child=SIGKILL",
            "--mount",
            "--mount-proc=/proc",
            scope.sys.executable,
            str(QUALIFICATION_DIR / "linux_namespace_exec.py"),
            "--",
            "/bin/true",
        ]
        with mock.patch.dict(
            os.environ,
            {
                scope.wsl_platform.NETWORK_ISOLATION_ENV: (
                    "util-linux-unshare-user-net-pid-landlock-v1"
                )
            },
            clear=True,
        ):
            scope._verify_network_command(command)
            with self.assertRaisesRegex(scope.ScopeExecError, "namespace boundary"):
                scope._verify_network_command(
                    [item for item in command if item != "--pid"]
                )
            with self.assertRaisesRegex(scope.ScopeExecError, "namespace boundary"):
                without_helper = list(command)
                without_helper[10] = "/tmp/untrusted.py"
                scope._verify_network_command(without_helper)

    def test_resource_arguments_reject_unenforceable_values(self) -> None:
        with contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit):
                scope.parse_args(
                    [
                        "--memory-max-bytes",
                        str(64 * 1024 * 1024),
                        "--pids-max",
                        "512",
                        "--cpu-quota-percent",
                        "101",
                        "--runtime-max-seconds",
                        "10",
                        "--",
                        "/bin/true",
                    ]
                )
        parsed = scope.parse_args(
            [
                "--memory-max-bytes",
                str(10 * 1024 * 1024 * 1024),
                "--pids-max",
                "512",
                "--cpu-quota-percent",
                "50",
                "--runtime-max-seconds",
                "30",
                "--",
                "/bin/true",
            ]
        )
        self.assertEqual(parsed.cpu_quota_percent, 50)

    def test_empty_host_build_inventory_is_accepted(self) -> None:
        with mock.patch.object(scope.Path, "iterdir", return_value=[]):
            self.assertEqual(scope._active_host_builds(), [])


if __name__ == "__main__":
    unittest.main()
