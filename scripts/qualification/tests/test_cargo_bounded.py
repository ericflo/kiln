from __future__ import annotations

import os
import subprocess
import tempfile
import threading
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
        self.assertIn("KILN_CARGO_MAX_MEMORY_GIB", completed.stdout)
        self.assertIn("closed-qualification-test-v1", completed.stdout)

    def test_qualification_launcher_pins_the_host_safety_contract(self) -> None:
        self.assertTrue(os.access(QUALIFICATION_SCRIPT, os.X_OK))
        source = QUALIFICATION_SCRIPT.read_text(encoding="utf-8")
        for assignment in (
            "CARGO_NET_OFFLINE=true",
            "KILN_CARGO_ENVIRONMENT_POLICY=closed-qualification-test-v1",
            "KILN_CARGO_EXECUTION_MODE=transient-service",
            "KILN_CARGO_JOBS=1",
            "KILN_CARGO_MIN_AVAILABLE_GIB=15",
            "KILN_CARGO_PRIVATE_NETWORK=1",
            "KILN_CARGO_SERVICE_RUNTIME_MAX_SECONDS=1740",
        ):
            self.assertIn(f"export {assignment}", source)
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
            (tool_dir / "systemctl").write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
            (tool_dir / "systemd-run").chmod(0o755)
            (tool_dir / "systemctl").chmod(0o755)
            environment = dict(os.environ)
            environment.update(
                {
                    "CARGO": "/bin/true",
                    "KILN_CARGO_EXECUTION_MODE": "transient-service",
                    "KILN_CARGO_MIN_AVAILABLE_GIB": "1",
                    "KILN_CARGO_PRIVATE_NETWORK": "1",
                    "KILN_CARGO_SERVICE_RUNTIME_MAX_SECONDS": "300",
                    "KILN_QUALIFICATION": "1",
                    "KILN_QUALIFICATION_HF_LOGITS_PATH": "/oracles/hf-logits.safetensors",
                    "KILN_QUALIFICATION_MODEL_PATH": "/models/source-pinned",
                    "KILN_TEST_SECRET_TOKEN": "must-not-enter-service",
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
            for expected in (
                "--wait",
                "--collect",
                "--pipe",
                "--setenv=PATH",
                "MemorySwapMax=0",
                "KillMode=control-group",
                "RuntimeMaxSec=300s",
                "PrivateNetwork=yes",
                "/bin/true",
                "check",
            ):
                self.assertIn(expected, arguments)
            self.assertNotIn("--setenv=KILN_TEST_SECRET_TOKEN", arguments)
            self.assertNotIn("--setenv=KILN_QUALIFICATION", arguments)
            self.assertNotIn("--setenv=KILN_QUALIFICATION_HF_LOGITS_PATH", arguments)
            self.assertNotIn("--setenv=KILN_QUALIFICATION_MODEL_PATH", arguments)

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
            self.assertIn("--setenv=KILN_QUALIFICATION", arguments)
            self.assertIn("--setenv=KILN_QUALIFICATION_HF_LOGITS_PATH", arguments)
            self.assertIn("--setenv=KILN_QUALIFICATION_MODEL_PATH", arguments)
            self.assertNotIn("--setenv=KILN_TEST_SECRET_TOKEN", arguments)

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
        self.assertIn("requires transient-service", completed.stderr)

    def test_transient_service_thermal_guard_stops_the_complete_build(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            tool_dir = root / "tools"
            tool_dir.mkdir()
            systemctl_args = root / "systemctl.args"
            (tool_dir / "systemd-run").write_text(
                "#!/bin/sh\nexec sleep 30\n",
                encoding="utf-8",
            )
            (tool_dir / "systemctl").write_text(
                "#!/bin/sh\nprintf '%s\\n' \"$@\" >> \"$KILN_TEST_SYSTEMCTL_ARGS\"\n",
                encoding="utf-8",
            )
            (tool_dir / "systemd-run").chmod(0o755)
            (tool_dir / "systemctl").chmod(0o755)

            hwmon = root / "hwmon" / "hwmon7"
            hwmon.mkdir(parents=True)
            (hwmon / "name").write_text("k10temp\n", encoding="utf-8")
            (hwmon / "temp1_label").write_text("Tctl\n", encoding="utf-8")
            temperature = hwmon / "temp1_input"
            temperature.write_text("40000\n", encoding="utf-8")

            def trip_sensor() -> None:
                time.sleep(0.2)
                replacement = hwmon / "temp1_input.next"
                replacement.write_text("97000\n", encoding="utf-8")
                os.replace(replacement, temperature)

            updater = threading.Thread(target=trip_sensor)
            updater.start()
            environment = dict(os.environ)
            environment.update(
                {
                    "CARGO": "/bin/true",
                    "KILN_CARGO_EXECUTION_MODE": "transient-service",
                    "KILN_CARGO_HOST_THERMAL_LIMIT_MILLICELSIUS": "97000",
                    "KILN_CARGO_HOST_THERMAL_POLL_MILLISECONDS": "50",
                    "KILN_CARGO_HOST_THERMAL_SENSOR_LABEL": "Tctl",
                    "KILN_CARGO_HOST_THERMAL_SENSOR_NAME": "k10temp",
                    "KILN_CARGO_HWMON_ROOT": str(root / "hwmon"),
                    "KILN_CARGO_MIN_AVAILABLE_GIB": "1",
                    "KILN_TEST_SYSTEMCTL_ARGS": str(systemctl_args),
                    "PATH": f"{tool_dir}:{environment['PATH']}",
                }
            )
            started = time.monotonic()
            completed = subprocess.run(
                [str(SCRIPT), "build"],
                cwd=ROOT,
                check=False,
                env=environment,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                timeout=5,
            )
            updater.join(timeout=1)
            self.assertEqual(completed.returncode, 3, completed.stderr)
            self.assertLess(time.monotonic() - started, 5)
            self.assertIn("thermal guard tripped", completed.stderr)
            self.assertIn("97000 millicelsius", completed.stderr)
            self.assertIn("stop", systemctl_args.read_text(encoding="utf-8"))

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
            empty_hwmon = root / "hwmon"
            empty_hwmon.mkdir()
            environment = dict(os.environ)
            environment.update(
                {
                    "CARGO": "/bin/true",
                    "KILN_CARGO_HWMON_ROOT": str(empty_hwmon),
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
