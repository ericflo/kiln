from __future__ import annotations

import importlib.util
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


MODULE_PATH = Path(__file__).resolve().parents[1] / "source_tree_hash.py"
SPEC = importlib.util.spec_from_file_location("source_tree_hash", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
source_tree_hash = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = source_tree_hash
SPEC.loader.exec_module(source_tree_hash)


class SourceTreeHashTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        subprocess.run(["git", "init", "-q"], cwd=self.root, check=True)
        self._write("Cargo.toml", "[workspace]\n")
        self._write("rust-toolchain.toml", "[toolchain]\nchannel = \"1.96.1\"\n")
        self._write("crates/demo/src/lib.rs", "pub fn value() -> u32 { 1 }\n")
        self._write("scripts/qualification/tool.py", "VALUE = 1\n")
        self._write("contracts/thinking-budget-v1.schema.json", "{}\n")
        self._write("qualification/schema/receipt.json", "{}\n")
        self._write("qualification/workloads/smoke.json", "{}\n")
        self._write("qualification/receipts/rocm/result.json", "{}\n")
        self._write("assets/logo.png", "runtime asset\n")
        self._write("assets/profiling/run.log", "historical evidence\n")
        self._write("desktop/src/main.rs", "fn main() {}\n")
        self._write("desktop/README.md", "desktop prose\n")
        self._write("scripts/c2_artifacts/reference.txt", "historical artifact\n")
        self._write("docs/plan.md", "not a runtime input\n")
        subprocess.run(["git", "add", "."], cwd=self.root, check=True)

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def _write(self, relative: str, content: str) -> None:
        path = self.root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content)

    def _hash(self) -> str:
        digest, _ = source_tree_hash.source_tree_hash(self.root)
        return digest

    def test_source_change_changes_hash(self) -> None:
        before = self._hash()
        self._write("crates/demo/src/lib.rs", "pub fn value() -> u32 { 2 }\n")
        self.assertNotEqual(before, self._hash())

    def test_contract_tool_schema_and_workload_change_hash(self) -> None:
        for relative in (
            "contracts/thinking-budget-v1.schema.json",
            "scripts/qualification/tool.py",
            "qualification/schema/receipt.json",
            "qualification/workloads/smoke.json",
            "rust-toolchain.toml",
        ):
            with self.subTest(relative=relative):
                before = self._hash()
                path = self.root / relative
                path.write_text(path.read_text() + "\n")
                self.assertNotEqual(before, self._hash())
                path.write_text(path.read_text()[:-1])

    def test_receipts_docs_and_historical_artifacts_do_not_change_hash(self) -> None:
        before = self._hash()
        self._write("qualification/receipts/rocm/result.json", '{"result":"passed"}\n')
        self._write("docs/plan.md", "updated plan\n")
        self._write("desktop/README.md", "updated desktop prose\n")
        self._write("assets/profiling/run.log", "updated historical evidence\n")
        self._write("scripts/c2_artifacts/reference.txt", "updated historical artifact\n")
        self.assertEqual(before, self._hash())

    def test_runtime_assets_and_desktop_source_change_hash(self) -> None:
        for relative in ("assets/logo.png", "desktop/src/main.rs"):
            with self.subTest(relative=relative):
                before = self._hash()
                path = self.root / relative
                original = path.read_text()
                path.write_text(original + "changed\n")
                self.assertNotEqual(before, self._hash())
                path.write_text(original)

    def test_input_order_is_deterministic(self) -> None:
        first_hash, first_entries = source_tree_hash.source_tree_hash(self.root)
        subprocess.run(["git", "add", "."], cwd=self.root, check=True)
        second_hash, second_entries = source_tree_hash.source_tree_hash(self.root)
        self.assertEqual(first_hash, second_hash)
        self.assertEqual(first_entries, second_entries)
        self.assertEqual(
            [entry.path for entry in first_entries],
            sorted(entry.path for entry in first_entries),
        )

    def test_executable_mode_changes_hash_after_index_update(self) -> None:
        before = self._hash()
        tool = self.root / "scripts/qualification/tool.py"
        tool.chmod(tool.stat().st_mode | 0o111)
        subprocess.run(["git", "add", str(tool)], cwd=self.root, check=True)
        self.assertNotEqual(before, self._hash())

    def test_unresolved_index_stage_is_rejected(self) -> None:
        relative = "crates/demo/src/lib.rs"
        oid = subprocess.check_output(
            ["git", "hash-object", "-w", relative], cwd=self.root, text=True
        ).strip()
        subprocess.run(
            ["git", "update-index", "--force-remove", relative],
            cwd=self.root,
            check=True,
        )
        subprocess.run(
            ["git", "update-index", "--index-info"],
            cwd=self.root,
            input=f"100644 {oid} 1\t{relative}\n",
            text=True,
            check=True,
        )

        with self.assertRaisesRegex(
            source_tree_hash.SourceTreeHashError,
            r"crates/demo/src/lib\.rs has unresolved index stage 1",
        ):
            self._hash()

    def test_symlink_hashes_target_text_without_following_target(self) -> None:
        first_target = self.root / "docs" / "first.py"
        second_target = self.root / "docs" / "second.py"
        self._write("docs/first.py", "VALUE = 1\n")
        self._write("docs/second.py", "VALUE = 2\n")
        link = self.root / "scripts" / "qualification" / "linked.py"
        link.symlink_to(os.path.relpath(first_target, link.parent))
        subprocess.run(["git", "add", "."], cwd=self.root, check=True)

        before = self._hash()
        first_target.write_text("VALUE = 99\n")
        self.assertEqual(before, self._hash())

        link.unlink()
        link.symlink_to(os.path.relpath(second_target, link.parent))
        self.assertNotEqual(before, self._hash())

    def test_gitlink_hashes_index_object_id_without_checkout(self) -> None:
        empty_tree = subprocess.check_output(
            ["git", "mktree"], cwd=self.root, input=b""
        ).decode("ascii").strip()
        commit_command = [
            "git",
            "-c",
            "user.name=Qualification Test",
            "-c",
            "user.email=qualification@example.invalid",
            "commit-tree",
            empty_tree,
        ]
        first_oid = subprocess.check_output(
            commit_command,
            cwd=self.root,
            input=b"first\n",
        ).decode("ascii").strip()
        relative = "crates/vendor"
        subprocess.run(
            [
                "git",
                "update-index",
                "--add",
                "--cacheinfo",
                f"160000,{first_oid},{relative}",
            ],
            cwd=self.root,
            check=True,
        )

        before = self._hash()
        second_oid = subprocess.check_output(
            [*commit_command, "-p", first_oid],
            cwd=self.root,
            input=b"second\n",
        ).decode("ascii").strip()
        subprocess.run(
            ["git", "update-index", "--cacheinfo", f"160000,{second_oid},{relative}"],
            cwd=self.root,
            check=True,
        )
        self.assertNotEqual(before, self._hash())

    def test_hash_uses_repository_sha256_convention(self) -> None:
        digest = self._hash()
        self.assertRegex(digest, r"^sha256:[0-9a-f]{64}$")


if __name__ == "__main__":
    unittest.main()
