from __future__ import annotations

import os
import subprocess
import tempfile
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


if __name__ == "__main__":
    unittest.main()
