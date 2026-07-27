from __future__ import annotations

import importlib.util
import contextlib
import io
import os
import stat
import subprocess
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
    def test_missing_runtime_unit_directory_is_created_with_private_mode(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            runtime = Path(directory)
            systemd_directory = runtime / "systemd"
            systemd_directory.mkdir(mode=0o700)

            unit_directory = scope._ensure_runtime_unit_directory(runtime)

            self.assertEqual(unit_directory, systemd_directory / "user")
            self.assertTrue(stat.S_ISDIR(unit_directory.lstat().st_mode))
            self.assertEqual(stat.S_IMODE(unit_directory.lstat().st_mode), 0o700)
            self.assertEqual(
                scope._ensure_runtime_unit_directory(runtime),
                unit_directory,
            )

    def test_runtime_unit_directory_rejects_symlink_and_writable_parent(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            runtime = Path(directory)
            systemd_directory = runtime / "systemd"
            systemd_directory.mkdir(mode=0o700)
            target = runtime / "target"
            target.mkdir(mode=0o700)
            (systemd_directory / "user").symlink_to(target, target_is_directory=True)
            with self.assertRaisesRegex(scope.ScopeExecError, "unsafe"):
                scope._ensure_runtime_unit_directory(runtime)

        with tempfile.TemporaryDirectory() as directory:
            runtime = Path(directory)
            systemd_directory = runtime / "systemd"
            systemd_directory.mkdir(mode=0o777)
            systemd_directory.chmod(0o777)
            with self.assertRaisesRegex(scope.ScopeExecError, "unsafe"):
                scope._ensure_runtime_unit_directory(runtime)

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
        unthrottled = scope.parse_args(
            [
                "--memory-max-bytes",
                "0",
                "--pids-max",
                "512",
                "--cpu-quota-percent",
                "0",
                "--runtime-max-seconds",
                "30",
                "--",
                "/bin/true",
            ]
        )
        self.assertEqual(unthrottled.cpu_quota_percent, 0)
        self.assertEqual(unthrottled.memory_max_bytes, 0)

    def test_empty_host_build_inventory_is_accepted(self) -> None:
        with mock.patch.object(scope.Path, "iterdir", return_value=[]):
            self.assertEqual(scope._active_host_builds(), [])

    def test_failed_transient_scope_is_reset_after_stop(self) -> None:
        failed = subprocess.CompletedProcess(
            args=[],
            returncode=0,
            stdout=b"failed\n",
            stderr=b"",
        )
        reset = subprocess.CompletedProcess(
            args=[],
            returncode=0,
            stdout=b"",
            stderr=b"",
        )
        with mock.patch.object(
            scope.subprocess,
            "run",
            side_effect=(failed, reset),
        ) as run:
            scope._reset_scope_failure("kiln-wsl-scope-fixture", {"A": "B"})

        self.assertEqual(
            run.call_args_list[1].args[0],
            [
                "/usr/bin/systemctl",
                "--user",
                "reset-failed",
                "kiln-wsl-scope-fixture.scope",
            ],
        )

    def test_inactive_transient_scope_needs_no_reset(self) -> None:
        inactive = subprocess.CompletedProcess(
            args=[],
            returncode=1,
            stdout=b"inactive\n",
            stderr=b"",
        )
        with mock.patch.object(
            scope.subprocess,
            "run",
            return_value=inactive,
        ) as run:
            scope._reset_scope_failure("kiln-wsl-scope-fixture", {"A": "B"})

        run.assert_called_once()

    def test_cgroup_freeze_waits_for_kernel_state_and_times_out_closed(self) -> None:
        cgroup = Path("/fixture/cgroup")
        with mock.patch.object(scope, "_write") as write, mock.patch.object(
            scope,
            "_read",
            return_value="1",
        ), mock.patch.object(
            scope,
            "_events",
            return_value={"frozen": 1},
        ):
            scope._set_frozen(cgroup, True)
        write.assert_called_once_with(cgroup / "cgroup.freeze", "1")

        with mock.patch.object(scope, "_write"), mock.patch.object(
            scope,
            "_read",
            return_value="1",
        ), mock.patch.object(
            scope,
            "_events",
            return_value={"frozen": 0},
        ), mock.patch.object(
            scope.time,
            "monotonic",
            side_effect=[0.0, 2.0],
        ):
            with self.assertRaisesRegex(scope.ScopeExecError, "did not freeze"):
                scope._set_frozen(cgroup, True)

if __name__ == "__main__":
    unittest.main()
