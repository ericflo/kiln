from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
MODULE_PATH = ROOT / "scripts" / "check_production_file_budget.py"
POLICY_PATH = ROOT / "contracts" / "production-file-budget-v1.json"
SPEC = importlib.util.spec_from_file_location("check_production_file_budget", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
budget = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = budget
SPEC.loader.exec_module(budget)


def source(path: str, lines: int) -> budget.SourceFile:
    return budget.SourceFile(path, lines)


class ProductionFileBudgetTests(unittest.TestCase):
    def test_checked_in_policy_and_current_tree_pass(self) -> None:
        policy = budget.load_policy(POLICY_PATH)
        files = budget.source_files(ROOT)
        self.assertEqual(policy.max_lines, 5000)
        self.assertEqual(len(policy.exceptions), 17)
        self.assertEqual(budget.violations(files, policy), [])

    def test_unlisted_growth_and_exception_growth_fail(self) -> None:
        configured = budget.Policy(
            100,
            (
                budget.ExceptionEntry(
                    "crates/demo/src/legacy.rs",
                    150,
                    "A deliberately long rationale that describes the ownership debt.",
                ),
            ),
        )
        errors = budget.violations(
            [
                source("crates/demo/src/legacy.rs", 151),
                source("crates/demo/src/new.rs", 101),
            ],
            configured,
        )
        self.assertEqual(len(errors), 2)
        self.assertIn("reviewed exception of 150", errors[0])
        self.assertIn("default budget of 100", errors[1])

    def test_stale_missing_and_headroom_exceptions_fail(self) -> None:
        configured = budget.Policy(
            100,
            (
                budget.ExceptionEntry("crates/demo/src/missing.rs", 120, "missing"),
                budget.ExceptionEntry("crates/demo/src/small.rs", 120, "small"),
                budget.ExceptionEntry("crates/demo/src/shrunk.rs", 120, "shrunk"),
            ),
        )
        errors = budget.violations(
            [
                source("crates/demo/src/small.rs", 100),
                source("crates/demo/src/shrunk.rs", 119),
            ],
            configured,
        )
        self.assertEqual(len(errors), 3)
        self.assertIn("missing or outside production scope", errors[0])
        self.assertIn("stale exception", errors[1])
        self.assertIn("exception ceiling has headroom", errors[2])

    def test_scope_includes_product_sources_but_excludes_test_children(self) -> None:
        self.assertTrue(budget.is_production_source(Path("crates/demo/src/lib.rs")))
        self.assertTrue(
            budget.is_production_source(Path("crates/kiln-server/src/ui/app.js"))
        )
        self.assertFalse(
            budget.is_production_source(Path("crates/demo/src/tests/mod.rs"))
        )
        self.assertFalse(budget.is_production_source(Path("scripts/tool.py")))
        self.assertFalse(budget.is_production_source(Path("crates/demo/tests/e2e.rs")))

    def test_policy_is_closed_sorted_and_requires_specific_rationale(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "policy.json"
            value = {
                "schema_version": 1,
                "max_production_file_lines": 100,
                "exceptions": [
                    {
                        "path": "crates/z/src/lib.rs",
                        "max_lines": 101,
                        "rationale": "This rationale is deliberately long enough to be specific.",
                    },
                    {
                        "path": "crates/a/src/lib.rs",
                        "max_lines": 101,
                        "rationale": "This rationale is also deliberately specific and complete.",
                    },
                ],
            }
            path.write_text(json.dumps(value))
            with self.assertRaisesRegex(budget.BudgetPolicyError, "sorted"):
                budget.load_policy(path)

            value["exceptions"] = [
                {
                    "path": "crates/a/src/lib.rs",
                    "max_lines": 101,
                    "rationale": "too short",
                }
            ]
            path.write_text(json.dumps(value))
            with self.assertRaisesRegex(budget.BudgetPolicyError, "at least 40"):
                budget.load_policy(path)

    def test_physical_line_count_handles_missing_terminal_newline(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "source.rs"
            path.write_bytes(b"one\ntwo")
            self.assertEqual(budget.physical_line_count(path), 2)
            path.write_bytes(b"one\ntwo\n")
            self.assertEqual(budget.physical_line_count(path), 2)


if __name__ == "__main__":
    unittest.main()
