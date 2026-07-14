from __future__ import annotations

import os
import subprocess
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
SCRIPT = ROOT / "scripts" / "cargo-bounded.sh"


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


if __name__ == "__main__":
    unittest.main()
