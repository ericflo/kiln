from __future__ import annotations

import importlib.util
import io
import json
import subprocess
import sys
import tempfile
import unittest
from contextlib import redirect_stderr
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
MODULE_PATH = ROOT / "scripts" / "check_repository_artifacts.py"
POLICY_PATH = ROOT / "contracts" / "repository-artifact-policy-v1.json"
SPEC = importlib.util.spec_from_file_location("check_repository_artifacts", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
artifacts = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = artifacts
SPEC.loader.exec_module(artifacts)


def policy(**overrides: object) -> artifacts.Policy:
    value = {
        "forbidden_suffixes": (".log", ".prom", ".sse"),
        "max_csv_bytes": 10,
        "max_file_bytes": 20,
        "exceptions": (),
    }
    value.update(overrides)
    return artifacts.Policy(**value)


def entry(path: str, size: int, oid: str = "1" * 40) -> artifacts.IndexedPath:
    return artifacts.IndexedPath(path, "100644", oid, "blob", size)


class RepositoryArtifactPolicyTests(unittest.TestCase):
    def test_checked_in_policy_is_closed_and_valid(self) -> None:
        loaded = artifacts.load_policy(POLICY_PATH)
        self.assertIn(".log", loaded.forbidden_suffixes)
        self.assertIn(".prom", loaded.forbidden_suffixes)
        self.assertLess(loaded.max_csv_bytes, loaded.max_file_bytes)

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "policy.json"
            path.write_text(
                POLICY_PATH.read_text().replace(
                    '"schema_version": 1,',
                    '"schema_version": 1,\n  "surprise": true,',
                )
            )
            with self.assertRaisesRegex(
                artifacts.ArtifactPolicyError, "unknown keys: surprise"
            ):
                artifacts.load_policy(path)

    def test_forbidden_suffixes_are_case_insensitive_and_nul_safe(self) -> None:
        values = [
            entry("audit.LOG", 1),
            entry("line\nbreak.sse", 1),
            entry("metrics.prom", 1),
            entry("summary.json", 1),
        ]
        violations = artifacts.find_violations(values, policy())
        self.assertEqual(
            [item.entry.path for item in violations],
            ["audit.LOG", "line\nbreak.sse", "metrics.prom"],
        )

    def test_csv_and_general_size_limits_are_independent(self) -> None:
        values = [
            entry("ok.csv", 10),
            entry("large.csv", 11),
            entry("ok.bin", 20),
            entry("large.bin", 21),
        ]
        violations = artifacts.find_violations(values, policy())
        self.assertEqual(
            [item.entry.path for item in violations], ["large.bin", "large.csv"]
        )
        csv = next(item for item in violations if item.entry.path == "large.csv")
        self.assertIn("CSV exceeds 10 bytes", csv.reasons)

    def test_large_file_exception_is_bound_to_path_size_and_hash(self) -> None:
        exception = artifacts.LargeFileException(
            "model.bin",
            21,
            "sha256:" + "a" * 64,
            "Required deterministic reference tensor.",
        )
        configured = policy(exceptions=(exception,))
        values = [entry("model.bin", 21)]
        self.assertEqual(
            artifacts.find_violations(values, configured, lambda _: exception.sha256), []
        )
        violations = artifacts.find_violations(
            values, configured, lambda _: "sha256:" + "b" * 64
        )
        self.assertEqual(len(violations), 1)
        self.assertTrue(
            any(
                "SHA-256 does not match" in reason
                for item in violations
                for reason in item.reasons
            )
        )

    def test_git_index_reader_preserves_spaces_and_newlines(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            subprocess.run(["git", "init", "-q"], cwd=root, check=True)
            for name in ("space name.log", "line\nbreak.sse", "summary.json"):
                (root / name).write_text(name)
            subprocess.run(["git", "add", "."], cwd=root, check=True)
            entries = artifacts.indexed_paths(root)
            self.assertEqual(
                [item.path for item in entries],
                ["line\nbreak.sse", "space name.log", "summary.json"],
            )
            violations = artifacts.find_violations(entries, policy())
            self.assertEqual(
                [item.entry.path for item in violations],
                ["line\nbreak.sse", "space name.log"],
            )

    def test_archive_records_indexed_hashes_and_restoration_contract(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            subprocess.run(["git", "init", "-q"], cwd=root, check=True)
            subprocess.run(
                ["git", "config", "user.email", "test@example.com"],
                cwd=root,
                check=True,
            )
            subprocess.run(
                ["git", "config", "user.name", "Test"], cwd=root, check=True
            )
            policy_path = root / "policy.json"
            policy_path.write_text(POLICY_PATH.read_text())
            (root / "raw log.log").write_bytes(b"raw evidence\n")
            subprocess.run(["git", "add", "."], cwd=root, check=True)
            subprocess.run(["git", "commit", "-qm", "fixture"], cwd=root, check=True)
            loaded = artifacts.load_policy(policy_path)
            entries = artifacts.indexed_paths(root)
            hasher = artifacts.WorktreeHasher(root)
            violations = artifacts.find_violations(entries, loaded, hasher.sha256)
            output = root / "archive.json"
            artifacts.write_archive(output, root, policy_path, violations, hasher)
            value = json.loads(output.read_text())
            self.assertEqual(value["summary"]["artifact_count"], 1)
            self.assertEqual(value["artifacts"][0]["path"], "raw log.log")
            self.assertRegex(
                value["artifacts"][0]["sha256"], r"^sha256:[0-9a-f]{64}$"
            )
            self.assertFalse(value["history_rewritten"])
            self.assertIn("git show", value["restoration_command"])

    def test_cli_refuses_policy_outside_root(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            outside = Path(tmp) / "policy.json"
            outside.write_text(POLICY_PATH.read_text())
            with redirect_stderr(io.StringIO()):
                result = artifacts.main(
                    ["--root", str(ROOT), "--policy", str(outside)]
                )
            self.assertEqual(result, 1)

    def test_current_index_satisfies_policy(self) -> None:
        loaded = artifacts.load_policy(POLICY_PATH)
        entries = artifacts.indexed_paths(ROOT)
        hasher = artifacts.WorktreeHasher(ROOT)
        self.assertEqual(
            artifacts.find_violations(entries, loaded, hasher.sha256), []
        )


if __name__ == "__main__":
    unittest.main()
