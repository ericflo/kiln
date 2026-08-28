from __future__ import annotations

import os
import subprocess
import tempfile
import time
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
SCRIPT = ROOT / "scripts" / "cargo-bounded.sh"
QUALIFICATION_SCRIPT = ROOT / "scripts" / "qualification" / "cargo-test-bounded.sh"
class BoundedCargoTests(unittest.TestCase):
    def test_help_is_side_effect_free(self) -> None:
        completed = subprocess.run(
            [str(SCRIPT), "--help"],
            cwd=ROOT,
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertIn("aggregate", completed.stdout)
        self.assertIn("CARGO", completed.stdout)
        self.assertIn("KILN_CARGO_CPU_QUOTA_PERCENT", completed.stdout)
        self.assertIn("KILN_CARGO_MAX_MEMORY_GIB", completed.stdout)
        self.assertIn("closed-qualification-test-v1", completed.stdout)

    def test_qualification_launcher_preserves_portable_safety_contract(self) -> None:
        self.assertTrue(os.access(QUALIFICATION_SCRIPT, os.X_OK))
        source = QUALIFICATION_SCRIPT.read_text(encoding="utf-8")
        for assignment in (
            "CARGO_NET_OFFLINE=true",
            "KILN_CARGO_ENVIRONMENT_POLICY=closed-qualification-test-v1",
            "KILN_CARGO_JOBS=1",
            "KILN_CARGO_PRIVATE_NETWORK=1",
            "KILN_CARGO_SERVICE_RUNTIME_MAX_SECONDS=1740",
        ):
            self.assertIn(f"export {assignment}", source)
        self.assertIn(
            'KILN_WSL2_SCOPE_BOUNDARY:-}" == "systemd-user-scope-feedback-v1"',
            source,
        )
        self.assertIn(
            'KILN_QUALIFICATION_NETWORK_ISOLATION:-}" == "macos-sandbox-loopback-only-v1"',
            source,
        )
        self.assertIn("export KILN_CARGO_EXECUTION_MODE=macos-contained", source)
        self.assertIn("export KILN_CARGO_EXECUTION_MODE=delegated-cgroup", source)
        self.assertIn("export KILN_CARGO_EXECUTION_MODE=transient-service", source)
        self.assertNotIn("export KILN_CARGO_MIN_AVAILABLE_GIB=", source)
        self.assertNotIn("export KILN_CARGO_CPU_QUOTA_PERCENT=", source)
        self.assertIn('exec scripts/cargo-bounded.sh "$@"', source)

    def test_memory_preflight_refuses_before_launching_cargo(self) -> None:
        environment = dict(os.environ)
        environment["KILN_CARGO_MIN_AVAILABLE_GIB"] = "999999"
        completed = subprocess.run(
            [str(SCRIPT), "check", "--version"],
            cwd=ROOT,
            check=False,
            env=environment,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        self.assertEqual(completed.returncode, 2, completed.stderr)
        self.assertIn("refusing Cargo", completed.stderr)
        self.assertIn("999999 GiB", completed.stderr)

    def test_explicit_cargo_path_is_accepted_before_memory_preflight(self) -> None:
        environment = dict(os.environ)
        environment["CARGO"] = "/bin/false"
        environment["KILN_CARGO_MIN_AVAILABLE_GIB"] = "999999"
        completed = subprocess.run(
            [str(SCRIPT), "check", "--version"],
            cwd=ROOT,
            check=False,
            env=environment,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        self.assertEqual(completed.returncode, 2, completed.stderr)
        self.assertIn("refusing Cargo", completed.stderr)
        self.assertNotIn("CARGO=", completed.stderr)

    def test_wrapper_refuses_to_use_itself_as_cargo(self) -> None:
        environment = dict(os.environ)
        environment["CARGO"] = str(SCRIPT)
        completed = subprocess.run(
            [str(SCRIPT), "check"],
            cwd=ROOT,
            check=False,
            env=environment,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        self.assertEqual(completed.returncode, 2, completed.stderr)
        self.assertIn("cannot point", completed.stderr)

    def test_transient_service_is_bounded_private_and_deadline_limited(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            tool_dir = Path(directory)
            arguments_path = tool_dir / "systemd-run.args"
            (tool_dir / "systemd-run").write_text(
                "#!/bin/sh\nprintf '%s\\n' \"$@\" > \"$KILN_TEST_SYSTEMD_RUN_ARGS\"\n",
                encoding="utf-8",
            )
            (tool_dir / "systemctl").write_text(
                "#!/bin/sh\nexit 0\n", encoding="utf-8"
            )
            (tool_dir / "systemd-run").chmod(0o755)
            (tool_dir / "systemctl").chmod(0o755)
            environment = dict(os.environ)
            environment.update(
                {
                    "CARGO": "/bin/true",
                    "CUDARC_CUDA_VERSION": "12080",
                    "KILN_CARGO_EXECUTION_MODE": "transient-service",
                    "KILN_CARGO_CPU_QUOTA_PERCENT": "50",
                    "KILN_CARGO_MIN_AVAILABLE_GIB": "1",
                    "KILN_CARGO_PRIVATE_NETWORK": "1",
                    "KILN_CARGO_SERVICE_RUNTIME_MAX_SECONDS": "300",
                    "KILN_CUDA_ARCHS": "80,89",
                    "KILN_QUALIFICATION": "1",
                    "KILN_QUALIFICATION_HF_LOGITS_PATH": "/oracles/hf-logits.safetensors",
                    "KILN_QUALIFICATION_MODEL_PATH": "/models/source-pinned",
                    "KILN_TEST_SECRET_TOKEN": "must-not-enter-service",
                    "KILN_TEST_SYSTEMD_RUN_ARGS": str(arguments_path),
                    "LC_ALL": "",
                    "PATH": f"{tool_dir}:{environment['PATH']}",
                }
            )
            completed = subprocess.run(
                [str(SCRIPT), "check"],
                cwd=ROOT,
                check=False,
                env=environment,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
            self.assertEqual(completed.returncode, 0, completed.stderr)
            self.assertIn("cpu_quota=50%", completed.stderr)
            arguments = arguments_path.read_text(encoding="utf-8").splitlines()
            for expected in (
                "--wait",
                "--collect",
                "--pipe",
                f"--setenv=PATH={environment['PATH']}",
                "--setenv=CUDARC_CUDA_VERSION=12080",
                "--setenv=KILN_CUDA_ARCHS=80,89",
                "--setenv=LC_ALL=",
                "CPUQuota=50%",
                "MemorySwapMax=0",
                "KillMode=control-group",
                "RuntimeMaxSec=300s",
                "PrivateNetwork=no",
                str(ROOT / "scripts/qualification/linux_namespace_exec.py"),
                "--map-root-user",
                "--kill-child=SIGKILL",
                "--mount-proc=/proc",
                "/bin/true",
                "check",
            ):
                self.assertIn(expected, arguments)
            self.assertFalse(
                any(
                    argument.startswith("--setenv=KILN_TEST_SECRET_TOKEN=")
                    for argument in arguments
                )
            )
            self.assertFalse(
                any(
                    argument.startswith("--setenv=KILN_QUALIFICATION=")
                    for argument in arguments
                )
            )
            self.assertFalse(
                any(
                    argument.startswith("--setenv=KILN_QUALIFICATION_HF_LOGITS_PATH=")
                    for argument in arguments
                )
            )
            self.assertFalse(
                any(
                    argument.startswith("--setenv=KILN_QUALIFICATION_MODEL_PATH=")
                    for argument in arguments
                )
            )

            environment["KILN_CARGO_ENVIRONMENT_POLICY"] = (
                "closed-qualification-test-v1"
            )
            completed = subprocess.run(
                [str(SCRIPT), "test"],
                cwd=ROOT,
                check=False,
                env=environment,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
            self.assertEqual(completed.returncode, 0, completed.stderr)
            arguments = arguments_path.read_text(encoding="utf-8").splitlines()
            self.assertIn("--setenv=KILN_QUALIFICATION=1", arguments)
            self.assertIn(
                "--setenv=KILN_QUALIFICATION_HF_LOGITS_PATH=/oracles/hf-logits.safetensors",
                arguments,
            )
            self.assertIn(
                "--setenv=KILN_QUALIFICATION_MODEL_PATH=/models/source-pinned",
                arguments,
            )
            self.assertFalse(
                any(
                    argument.startswith("--setenv=KILN_TEST_SECRET_TOKEN=")
                    for argument in arguments
                )
            )

    def test_scope_applies_aggregate_cpu_quota(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            tool_dir = root / "tools"
            tool_dir.mkdir()
            arguments_path = root / "systemd-run.args"
            (tool_dir / "systemd-run").write_text(
                "#!/bin/sh\nprintf '%s\\n' \"$@\" > \"$KILN_TEST_SYSTEMD_RUN_ARGS\"\n",
                encoding="utf-8",
            )
            (tool_dir / "systemctl").write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
            (tool_dir / "systemd-run").chmod(0o755)
            (tool_dir / "systemctl").chmod(0o755)
            environment = dict(os.environ)
            environment.update(
                {
                    "CARGO": "/bin/true",
                    "KILN_CARGO_CPU_QUOTA_PERCENT": "250",
                    "KILN_CARGO_MIN_AVAILABLE_GIB": "1",
                    "KILN_TEST_SYSTEMD_RUN_ARGS": str(arguments_path),
                    "PATH": f"{tool_dir}:{environment['PATH']}",
                }
            )
            completed = subprocess.run(
                [str(SCRIPT), "check"],
                cwd=ROOT,
                check=False,
                env=environment,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
            self.assertEqual(completed.returncode, 0, completed.stderr)
            arguments = arguments_path.read_text(encoding="utf-8").splitlines()
            self.assertIn("--scope", arguments)
            self.assertIn("CPUQuota=250%", arguments)
            self.assertIn("cpu_quota=250%", completed.stderr)

    def test_invalid_cpu_quota_is_rejected_before_launch(self) -> None:
        environment = dict(os.environ)
        environment.update(
            {
                "CARGO": "/bin/true",
                "KILN_CARGO_CPU_QUOTA_PERCENT": "0",
                "KILN_CARGO_MIN_AVAILABLE_GIB": "1",
            }
        )
        completed = subprocess.run(
            [str(SCRIPT), "check"],
            cwd=ROOT,
            check=False,
            env=environment,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        self.assertEqual(completed.returncode, 2, completed.stderr)
        self.assertIn("must be in 1..=10000", completed.stderr)

    def test_private_network_requires_transient_service(self) -> None:
        environment = dict(os.environ)
        environment.update(
            {
                "CARGO": "/bin/true",
                "KILN_CARGO_MIN_AVAILABLE_GIB": "1",
                "KILN_CARGO_PRIVATE_NETWORK": "1",
            }
        )
        completed = subprocess.run(
            [str(SCRIPT), "check"],
            cwd=ROOT,
            check=False,
            env=environment,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        self.assertEqual(completed.returncode, 2, completed.stderr)
        self.assertIn("requires a contained execution mode", completed.stderr)

    @unittest.skipUnless(os.uname().sysname == "Darwin", "requires macOS sandbox")
    def test_macos_contained_mode_inherits_verified_boundary(self) -> None:
        environment = dict(os.environ)
        environment.update(
            {
                "CARGO": "/usr/bin/true",
                "KILN_CARGO_ENVIRONMENT_POLICY": "closed-qualification-test-v1",
                "KILN_CARGO_EXECUTION_MODE": "macos-contained",
                "KILN_CARGO_MIN_AVAILABLE_GIB": "1",
                "KILN_CARGO_PRIVATE_NETWORK": "1",
                "KILN_QUALIFICATION": "1",
                "KILN_QUALIFICATION_NETWORK_ISOLATION": (
                    "macos-sandbox-loopback-only-v1"
                ),
            }
        )
        profile = """(version 1)
(allow default)
(deny network-inbound)
(deny network-outbound)
(allow network-inbound (local ip "localhost:*"))
(allow network-outbound (remote ip "localhost:*"))
"""
        completed = subprocess.run(
            ["sandbox-exec", "-p", profile, str(SCRIPT), "check"],
            cwd=ROOT,
            check=False,
            env=environment,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            start_new_session=True,
        )
        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertIn("mode=macos-contained", completed.stderr)
        self.assertIn("aggregate_limit=unavailable", completed.stderr)
        self.assertIn("swap_limit=unavailable", completed.stderr)

    def test_cancelled_scope_stops_its_named_unit_and_runner(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            tool_dir = root / "tools"
            tool_dir.mkdir()
            runner_pid_path = root / "runner.pid"
            systemctl_args = root / "systemctl.args"
            (tool_dir / "systemd-run").write_text(
                "#!/bin/sh\nprintf '%s\\n' \"$$\" > \"$KILN_TEST_RUNNER_PID\"\nexec sleep 30\n",
                encoding="utf-8",
            )
            (tool_dir / "systemctl").write_text(
                "#!/bin/sh\nprintf '%s\\n' \"$@\" >> \"$KILN_TEST_SYSTEMCTL_ARGS\"\n",
                encoding="utf-8",
            )
            (tool_dir / "systemd-run").chmod(0o755)
            (tool_dir / "systemctl").chmod(0o755)
            environment = dict(os.environ)
            environment.update(
                {
                    "CARGO": "/bin/true",
                    "KILN_CARGO_MIN_AVAILABLE_GIB": "1",
                    "KILN_TEST_RUNNER_PID": str(runner_pid_path),
                    "KILN_TEST_SYSTEMCTL_ARGS": str(systemctl_args),
                    "PATH": f"{tool_dir}:{environment['PATH']}",
                }
            )
            process = subprocess.Popen(
                [str(SCRIPT), "check"],
                cwd=ROOT,
                env=environment,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
            deadline = time.monotonic() + 2
            while not runner_pid_path.is_file() and time.monotonic() < deadline:
                time.sleep(0.01)
            self.assertTrue(runner_pid_path.is_file())
            runner_pid = int(runner_pid_path.read_text(encoding="utf-8"))
            process.terminate()
            process.communicate(timeout=3)
            self.assertIn("stop", systemctl_args.read_text(encoding="utf-8"))
            with self.assertRaises(ProcessLookupError):
                os.kill(runner_pid, 0)


if __name__ == "__main__":
    unittest.main()
