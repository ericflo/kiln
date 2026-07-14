from __future__ import annotations

import hashlib
import importlib.util
import json
import subprocess
import sys
import tempfile
import unittest
from contextlib import contextmanager
from pathlib import Path
from unittest import mock


REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
SCRIPTS_DIR = REPOSITORY_ROOT / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))

CHECKER_SPEC = importlib.util.spec_from_file_location(
    "backend_latency_fixture_checker",
    SCRIPTS_DIR / "check_backend_latency_fixtures.py",
)
assert CHECKER_SPEC is not None and CHECKER_SPEC.loader is not None
checker = importlib.util.module_from_spec(CHECKER_SPEC)
sys.modules[CHECKER_SPEC.name] = checker
CHECKER_SPEC.loader.exec_module(checker)

import write_backend_latency_result_artifact as writer


RAW_BYTES = (
    b"KILN_LATENCY_METRIC latency_ms 9.5 ms\n"
    b"KILN_LATENCY_METRIC tokens_per_s 125.0 tok/s\n"
)


def run_git(root: Path, *args: str) -> str:
    return subprocess.run(
        ["git", *args],
        cwd=root,
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    ).stdout.strip()


@contextmanager
def latency_roots(root: Path):
    with mock.patch.object(checker, "ROOT", root), mock.patch.object(
        writer, "ROOT", root
    ):
        yield


class BackendLatencyArtifactContractTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        run_git(self.root, "init", "-q")
        run_git(self.root, "config", "user.email", "test@example.com")
        run_git(self.root, "config", "user.name", "Test")

        (self.root / ".gitignore").write_text("*.log\n")
        self.source = self.root / "bench.py"
        self.source.write_text("print('latency fixture')\n")
        run_git(self.root, "add", ".gitignore", "bench.py")
        run_git(self.root, "commit", "-qm", "fixture source")
        source_commit = run_git(self.root, "rev-parse", "HEAD")

        self.raw_path = Path("bench-results/backend-latency/raw/fixture.log")
        self.result_path = Path("bench-results/backend-latency/fixture.json")
        self.manifest_path = Path("docs/fixtures.json")
        self.fixture = {
            "id": "latency_fixture",
            "backend": "rocm",
            "hardware": "test device",
            "source": "bench.py",
            "command": "python3 bench.py",
            "result_artifact": str(self.result_path),
            "threshold_state": "locked_threshold",
            "metrics": [
                {
                    "name": "latency_ms",
                    "unit": "ms",
                    "comparison": "<=",
                    "max": 10.0,
                },
                {
                    "name": "tokens_per_s",
                    "unit": "tok/s",
                    "comparison": ">=",
                    "max": 100.0,
                },
            ],
        }
        self.result = {
            "artifact_schema_version": writer.ARTIFACT_SCHEMA_VERSION,
            "backend": "rocm",
            "command": "python3 bench.py",
            "created_at_utc": "2026-07-13T12:00:00Z",
            "fixture_id": "latency_fixture",
            "fixture_spec_sha256": writer.fixture_spec_sha256(self.fixture),
            "git_commit": source_commit,
            "git_tracked_dirty": False,
            "hardware": "test device",
            "manifest": str(self.manifest_path),
            "manifest_schema_version": 1,
            "metrics": {"latency_ms": 9.5, "tokens_per_s": 125.0},
            "raw_log": str(self.raw_path),
            "raw_log_sha256": hashlib.sha256(RAW_BYTES).hexdigest(),
            "source": "bench.py",
            "source_sha256": hashlib.sha256(self.source.read_bytes()).hexdigest(),
            "status": "passed",
        }
        self.manifest = {
            "schema_version": 1,
            "status": "covered",
            "policy": {
                "covered_gate_requires": checker.REQUIRED_COVERED_GATE_POLICY,
            },
            "required_backends": ["rocm"],
            "fixtures": [self.fixture],
            "missing_fixture_slots": [],
        }

        result = self.root / self.result_path
        result.parent.mkdir(parents=True)
        result.write_text(json.dumps(self.result))
        manifest = self.root / self.manifest_path
        manifest.parent.mkdir(parents=True)
        manifest.write_text(json.dumps(self.manifest))
        run_git(self.root, "add", str(self.result_path), str(self.manifest_path))
        run_git(self.root, "commit", "-qm", "compact latency evidence")

    def tearDown(self) -> None:
        self.temp.cleanup()

    def validate(self) -> list[str]:
        with latency_roots(self.root):
            return checker.validate_manifest(
                self.manifest,
                require_covered=True,
                manifest_path=self.root / self.manifest_path,
            )

    def test_covered_gate_accepts_compact_evidence_without_raw_log(self) -> None:
        self.assertFalse((self.root / self.raw_path).exists())
        self.assertEqual(self.validate(), [])

    def test_present_raw_log_is_rehashed_and_reparsed(self) -> None:
        raw_log = self.root / self.raw_path
        raw_log.parent.mkdir(parents=True)
        raw_log.write_bytes(RAW_BYTES.replace(b"9.5", b"12.0"))
        errors = self.validate()
        self.assertTrue(any("raw_log_sha256 does not match" in error for error in errors))
        self.assertTrue(any("must match raw_log value 12.0" in error for error in errors))

    def test_tracked_raw_log_is_rejected_even_when_checkout_file_is_absent(self) -> None:
        raw_log = self.root / self.raw_path
        raw_log.parent.mkdir(parents=True)
        raw_log.write_bytes(RAW_BYTES)
        run_git(self.root, "add", "-f", str(self.raw_path))
        run_git(self.root, "commit", "-qm", "incorrectly track raw log")
        raw_log.unlink()

        errors = self.validate()
        self.assertTrue(any("raw_log must not be tracked by git" in error for error in errors))

    def test_writer_clean_marker_ignores_raw_logs_but_not_other_untracked_files(self) -> None:
        raw_log = self.root / self.raw_path
        raw_log.parent.mkdir(parents=True)
        raw_log.write_bytes(RAW_BYTES)
        with latency_roots(self.root):
            self.assertFalse(writer.tracked_git_dirty())
            (self.root / "unexpected.txt").write_text("untracked\n")
            self.assertTrue(writer.tracked_git_dirty())


if __name__ == "__main__":
    unittest.main()
