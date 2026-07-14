from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
MODULE_PATH = ROOT / "scripts" / "check_source_parsing_tests.py"
SPEC = importlib.util.spec_from_file_location("check_source_parsing_tests", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
inventory = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = inventory
SPEC.loader.exec_module(inventory)


class SourceParsingInventoryTests(unittest.TestCase):
    def rust_entries(self, source: str) -> list[dict[str, object]]:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            test_path = root / "crates" / "demo" / "tests" / "contract.rs"
            test_path.parent.mkdir(parents=True)
            test_path.write_text(source)
            return inventory.scan_rust(root)

    def test_rust_comments_strings_and_fixture_reads_are_excluded(self) -> None:
        entries = self.rust_entries(
            r'''
const FIXTURE: &str = include_str!("fixtures/reference.rs");
// const DECOY: &str = include_str!("../src/decoy.rs");

#[test]
fn structured_fixture_is_not_implementation_source() {
    let decoy = "fs::read_to_string(root.join(\"src/decoy.rs\"))";
    assert!(FIXTURE.contains("record"));
    assert!(decoy.contains("decoy"));
}
'''
        )
        self.assertEqual(entries, [])

    def test_module_binding_propagates_through_async_helper_chain(self) -> None:
        entries = self.rust_entries(
            r'''
const IMPLEMENTATION: &str = include_str!("../src/lib.rs");

fn implementation() -> &'static str { IMPLEMENTATION }
fn section() -> &'static str { implementation() }

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn helper_chain_is_inventoried() {
    assert!(section().contains("contract"));
}
'''
        )
        self.assertEqual(len(entries), 1)
        self.assertEqual(entries[0]["test_name"], "helper_chain_is_inventoried")
        self.assertEqual(
            entries[0]["read_sites"],
            [
                {
                    "api": "module_include_str",
                    "target": "crates/demo/src/lib.rs",
                }
            ],
        )
        self.assertEqual(entries[0]["text_assertion_count"], 1)

    def test_direct_implementation_read_is_classified(self) -> None:
        entries = self.rust_entries(
            r'''
#[test]
fn direct_read_is_inventoried() {
    let source = std::fs::read_to_string(root.join("crates/demo/src/lib.rs")).unwrap();
    assert!(source.contains("contract"));
}
'''
        )
        self.assertEqual(len(entries), 1)
        self.assertEqual(entries[0]["classification"], "implementation_source_text")
        self.assertEqual(entries[0]["read_sites"][0]["api"], "read_to_string")

    def test_workspace_reader_helper_indirection_is_classified(self) -> None:
        entries = self.rust_entries(
            r'''
fn workspace_root() -> PathBuf { PathBuf::from("workspace") }
fn read(path: &str) -> String {
    std::fs::read_to_string(workspace_root().join(path)).unwrap()
}

#[test]
fn helper_read_is_inventoried() {
    let source = read("crates/demo/src/lib.rs");
    assert!(source.contains("contract"));
}
'''
        )
        self.assertEqual(len(entries), 1)
        self.assertEqual(entries[0]["test_name"], "helper_read_is_inventoried")
        self.assertEqual(entries[0]["classification"], "implementation_source_text")

    def test_dynamic_path_read_uses_implementation_targets_from_test_body(self) -> None:
        entries = self.rust_entries(
            r'''
#[test]
fn dynamic_reads_are_inventoried() {
    for relative in ["crates/demo/src/lib.rs", "crates/demo/src/runtime.rs"] {
        let path = root.join(relative);
        let source = std::fs::read_to_string(&path).unwrap();
        assert!(source.contains("contract"));
    }
}
'''
        )
        self.assertEqual(len(entries), 1)
        self.assertEqual(
            entries[0]["read_sites"][0]["target"],
            "crates/demo/src/lib.rs, crates/demo/src/runtime.rs",
        )

    def test_generated_output_reader_is_not_misclassified(self) -> None:
        entries = self.rust_entries(
            r'''
#[test]
fn generated_output_is_behavioral_evidence() {
    let output = std::fs::read_to_string(root.join("target/result.json")).unwrap();
    assert!(output.contains("passed"));
}
'''
        )
        self.assertEqual(entries, [])

    def test_python_driver_read_is_classified(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            path = root / "scripts" / "qualification" / "tests" / "test_driver.py"
            path.parent.mkdir(parents=True)
            path.write_text(
                '''
def test_driver_contract():
    source = (qualification_dir / "serve_rocm.py").read_text()
    assert "typed_config" in source
'''
            )
            entries = inventory.scan_python(root)
        self.assertEqual(len(entries), 1)
        self.assertEqual(entries[0]["classification"], "qualification_driver_source_text")
        self.assertEqual(entries[0]["text_assertion_count"], 1)

    def test_ratchet_rejects_increases_and_lowers_all_observed_counts(self) -> None:
        value = {
            "summary": {
                "test_count": 3,
                "read_site_count": 5,
                "text_assertion_count": 7,
            },
            "ratchet": {
                "max_test_count": 2,
                "max_read_site_count": 5,
                "max_text_assertion_count": 8,
            },
        }
        with self.assertRaisesRegex(inventory.InventoryError, "test_count is 3"):
            inventory.enforce_ratchet(value)

        value["ratchet"]["max_test_count"] = 4
        inventory.enforce_ratchet(value)
        self.assertEqual(
            inventory.lowered_ratchet(value),
            {
                "max_test_count": 3,
                "max_read_site_count": 5,
                "max_text_assertion_count": 7,
            },
        )

    def test_checked_in_contract_and_report_are_exact(self) -> None:
        expected = inventory.load_contract(
            ROOT / "contracts" / "source-parsing-test-inventory-v1.json"
        )
        actual = inventory.build_inventory(ROOT, ratchet=expected["ratchet"])
        inventory.enforce_ratchet(actual)
        self.assertEqual(actual, expected)
        self.assertEqual(
            inventory.render_report(actual),
            (ROOT / "docs" / "VERIFICATION_TEST_INVENTORY.md").read_text(),
        )
        self.assertEqual(
            json.loads(json.dumps(actual, sort_keys=True)),
            json.loads(json.dumps(expected, sort_keys=True)),
        )


if __name__ == "__main__":
    unittest.main()
