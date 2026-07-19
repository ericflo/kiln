from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


QUALIFICATION_DIR = Path(__file__).resolve().parents[1]
if str(QUALIFICATION_DIR) not in sys.path:
    sys.path.insert(0, str(QUALIFICATION_DIR))

import guarded_exec


class GuardedExecTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.executable = self.root / "worker"
        self.executable.write_bytes(b"#!/bin/sh\nexit 0\n")
        self.executable.chmod(0o700)
        self.spec = self.root / "spec.json"
        self.value = {
            "argv": [str(self.executable), "--fixed", "value"],
            "cwd": str(self.root),
            "environment": {"LANG": "C.UTF-8", "PATH": "/usr/bin:/bin"},
            "executable": {
                "path": str(self.executable),
                "sha256": "sha256:" + hashlib.sha256(self.executable.read_bytes()).hexdigest(),
            },
            "schema": guarded_exec.SCHEMA,
        }

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def write_spec(self) -> None:
        self.spec.write_text(json.dumps(self.value), encoding="ascii")

    def test_spec_is_closed_hash_bound_and_exact(self) -> None:
        self.write_spec()
        executable, argv, cwd, environment, digest = guarded_exec.load_spec(self.spec)
        self.assertEqual(executable, self.executable)
        self.assertEqual(argv, self.value["argv"])
        self.assertEqual(cwd, self.root)
        self.assertEqual(environment, self.value["environment"])
        self.assertEqual(digest, self.value["executable"]["sha256"])

        self.value["unexpected"] = True
        self.write_spec()
        with self.assertRaisesRegex(guarded_exec.GuardedExecError, "fields or schema"):
            guarded_exec.load_spec(self.spec)

    def test_hash_and_argv_drift_fail_before_release(self) -> None:
        self.write_spec()
        self.executable.write_bytes(b"#!/bin/sh\nexit 1\n")
        with self.assertRaisesRegex(guarded_exec.GuardedExecError, "hash"):
            guarded_exec.load_spec(self.spec)

        self.value["executable"]["sha256"] = (
            "sha256:" + hashlib.sha256(self.executable.read_bytes()).hexdigest()
        )
        self.value["argv"][0] = "/wrong/executable"
        self.write_spec()
        with self.assertRaisesRegex(guarded_exec.GuardedExecError, r"argv\[0\]"):
            guarded_exec.load_spec(self.spec)

    def test_start_gate_is_private_regular_and_exact(self) -> None:
        gate = self.root / "start.gate"
        gate.write_bytes(guarded_exec.GATE_PAYLOAD)
        gate.chmod(0o600)
        guarded_exec.wait_for_gate(gate, timeout_seconds=0.01)

        gate.write_bytes(b"wrong")
        with self.assertRaisesRegex(guarded_exec.GuardedExecError, "payload"):
            guarded_exec.wait_for_gate(gate, timeout_seconds=0.01)
        gate.write_bytes(guarded_exec.GATE_PAYLOAD)
        gate.chmod(0o644)
        with self.assertRaisesRegex(guarded_exec.GuardedExecError, "permissions"):
            guarded_exec.wait_for_gate(gate, timeout_seconds=0.01)

    def test_open_descriptor_remains_bound_to_the_validated_inode(self) -> None:
        self.write_spec()
        expected = self.value["executable"]["sha256"]
        descriptor = guarded_exec.open_hash_bound_executable(self.executable, expected)
        try:
            replacement = self.root / "replacement"
            replacement.write_bytes(b"#!/bin/sh\nexit 1\n")
            replacement.chmod(0o700)
            os.replace(replacement, self.executable)
            self.assertEqual(guarded_exec._sha256_descriptor(descriptor), expected)
            self.assertNotEqual(guarded_exec._sha256(self.executable), expected)
        finally:
            os.close(descriptor)

    def test_cli_executes_the_admitted_descriptor_after_release(self) -> None:
        executable = Path("/usr/bin/true").resolve(strict=True)
        self.value["argv"] = [str(executable)]
        self.value["executable"] = {
            "path": str(executable),
            "sha256": guarded_exec._sha256(executable),
        }
        self.write_spec()
        gate = self.root / "start.gate"
        gate.write_bytes(guarded_exec.GATE_PAYLOAD)
        gate.chmod(0o600)
        completed = subprocess.run(
            [
                sys.executable,
                str(Path(guarded_exec.__file__).resolve(strict=True)),
                "--spec",
                str(self.spec),
                "--start-gate",
                str(gate),
            ],
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=10,
        )
        self.assertEqual(completed.returncode, 0, completed.stderr)


if __name__ == "__main__":
    unittest.main()
