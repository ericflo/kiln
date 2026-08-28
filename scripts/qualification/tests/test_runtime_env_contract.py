from __future__ import annotations

import importlib.util
import io
import json
import sys
import tempfile
import unittest
from contextlib import redirect_stderr
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
MODULE_PATH = ROOT / "scripts" / "check_runtime_env_contract.py"
CONTRACT_PATH = ROOT / "contracts" / "runtime-env-direct-reads-v1.json"
REPORT_PATH = ROOT / "docs" / "contracts" / "RUNTIME_ENVIRONMENT_INVENTORY.md"
SPEC = importlib.util.spec_from_file_location("check_runtime_env_contract", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
runtime_env = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = runtime_env
SPEC.loader.exec_module(runtime_env)


class RuntimeEnvironmentContractTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def _write(self, relative: str, content: str) -> None:
        path = self.root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")

    def _write_closed_runtime_boundaries(self) -> None:
        self._write(
            "crates/kiln-memory/src/startup_environment.rs",
            "pub fn capture(name: &str) { let _ = std::env::var_os(name); }\n",
        )
        self._write(
            "crates/kiln-train/src/credential_provider.rs",
            "pub fn resolve(name: &str) { let _ = std::env::var(name); }\n",
        )

    @staticmethod
    def _calls(contract: dict[str, object], section: str) -> set[tuple[str, str, str, int]]:
        entries = contract[section]
        assert isinstance(entries, list)
        return {
            (str(entry["api"]), str(entry["argument_kind"]), str(entry["argument"]), int(entry["count"]))
            for entry in entries
        }

    @staticmethod
    def _entry(
        contract: dict[str, object], section: str, argument: str
    ) -> dict[str, object]:
        entries = contract[section]
        assert isinstance(entries, list)
        return next(entry for entry in entries if entry["argument"] == argument)

    def test_rust_scan_ignores_comments_and_string_contents(self) -> None:
        self._write(
            "crates/demo/src/lib.rs",
            r'''
const NORMAL_DECOY: &str = "std::env::var(\"KILN_DECOY\")";
const RAW_DECOY: &str = r#"std::env::set_var("KILN_DECOY", "1")"#;
// std::env::var("KILN_COMMENT_DECOY");
/* std::env::remove_var("KILN_BLOCK_DECOY"); */
fn actual() {
    let _ = std::env::var("KILN_REAL");
}
''',
        )
        contract = runtime_env.build_contract(self.root)
        self.assertEqual(
            self._calls(contract, "reads"),
            {("var", "literal", "KILN_REAL", 1)},
        )
        self.assertEqual(self._calls(contract, "process_mutations"), set())

    def test_rust_aliases_helpers_macros_and_mutations_are_normalized(self) -> None:
        self._write(
            "crates/demo/src/lib.rs",
            '''
use std::env as process_env;
use std::env::{remove_var as clear_one, var_os as lookup};

fn env_flag(_name: &str, _default: bool) -> bool { false }
fn inspect(name: &str) {
    let _ = process_env::var("KILN_LITERAL");
    let _ = lookup(name);
    let _ = kiln_core::env_flag::env_flag("KILN_FLAG", false);
    let _ = kiln_core::env_flag::env_tristate(name);
    let _ = env!("CARGO_MANIFEST_DIR");
    let _ = option_env!("KILN_COMPILE_OPTION");
    unsafe { process_env::set_var("KILN_MUTATED", "1"); }
    unsafe { clear_one("KILN_REMOVED"); }
}
''',
        )
        contract = runtime_env.build_contract(self.root)
        reads = self._calls(contract, "reads")
        self.assertIn(("var", "literal", "KILN_LITERAL", 1), reads)
        self.assertIn(("var_os", "expression", "name", 1), reads)
        self.assertIn(("env_flag", "literal", "KILN_FLAG", 1), reads)
        self.assertIn(("env_tristate", "expression", "name", 1), reads)
        self.assertIn(("env!", "literal", "CARGO_MANIFEST_DIR", 1), reads)
        self.assertIn(("option_env!", "literal", "KILN_COMPILE_OPTION", 1), reads)
        self.assertNotIn(("env_flag", "expression", "_name", 1), reads)
        self.assertEqual(
            self._calls(contract, "process_mutations"),
            {
                ("set_var", "literal", "KILN_MUTATED", 1),
                ("remove_var", "literal", "KILN_REMOVED", 1),
            },
        )

    def test_native_scan_tracks_literal_and_dynamic_getenv_without_decorations(self) -> None:
        self._write(
            "crates/demo/csrc/kernel.cpp",
            r'''
const char* normal_deocy = "std::getenv(\"KILN_DECOY\")";
const char* raw_decoy = R"tag(std::getenv("KILN_RAW_DECOY"))tag";
// std::getenv("KILN_COMMENT_DECOY");
const char* read_literal() { return std::getenv("KILN_NATIVE"); }
const char* read_dynamic(const char* name) { return getenv(name); }
''',
        )
        contract = runtime_env.build_contract(self.root)
        self.assertEqual(
            self._calls(contract, "reads"),
            {
                ("getenv", "literal", "KILN_NATIVE", 1),
                ("getenv", "expression", "name", 1),
            },
        )

    def test_counts_are_deterministic_and_ignore_line_number_churn(self) -> None:
        relative = "crates/demo/src/lib.rs"
        source = '''
fn read() {
    let _ = std::env::var("KILN_DUPLICATE");
    let _ = std::env::var("KILN_DUPLICATE");
}
'''
        self._write(relative, source)
        before = runtime_env.build_contract(self.root)
        self._write(relative, "\n// moved down without changing behavior\n\n" + source)
        after = runtime_env.build_contract(self.root)
        self.assertEqual(before, after)
        self.assertIn(("var", "literal", "KILN_DUPLICATE", 2), self._calls(after, "reads"))

    def test_compile_time_macro_only_file_survives_prefilter(self) -> None:
        self._write(
            "crates/demo/src/lib.rs",
            'const ROOT: &str = env!("CARGO_MANIFEST_DIR");\n',
        )
        self.assertEqual(
            self._calls(runtime_env.build_contract(self.root), "reads"),
            {("env!", "literal", "CARGO_MANIFEST_DIR", 1)},
        )

    def test_unit_test_accesses_are_not_classified_as_production_runtime(self) -> None:
        self._write(
            "crates/demo/src/lib.rs",
            '''
const PROD_ENV: &str = "KILN_PRODUCTION_EXPERIMENT";

pub fn production() {
    let _ = std::env::var(PROD_ENV);
}

#[cfg(test)]
mod tests {
    #[test]
    fn isolated() {
        let _ = std::env::var("KILN_UNIT_READ");
        unsafe { std::env::set_var("KILN_UNIT_MUTATION", "1"); }
    }
}
''',
        )
        contract = runtime_env.build_contract(self.root)
        production = self._entry(contract, "reads", "KILN_PRODUCTION_EXPERIMENT")
        unit_read = self._entry(contract, "reads", "KILN_UNIT_READ")
        unit_mutation = self._entry(
            contract, "process_mutations", "KILN_UNIT_MUTATION"
        )
        self.assertEqual(production["surface"], "source")
        self.assertEqual(production["classification"], "experimental_debug")
        self.assertEqual(unit_read["surface"], "unit-test")
        self.assertEqual(unit_read["classification"], "test_only")
        self.assertEqual(unit_mutation["surface"], "unit-test")
        self.assertEqual(unit_mutation["classification"], "test_only")

    def test_public_build_and_integration_surfaces_have_distinct_ownership(self) -> None:
        self._write(
            "crates/kiln-server/src/config.rs",
            'pub fn load() { let _ = std::env::var("KILN_SERVER_PORT"); }\n',
        )
        self._write(
            "crates/demo/build.rs",
            'fn main() { let _ = std::env::var("KILN_DEMO_ARCH"); }\n',
        )
        self._write(
            "crates/demo/tests/device.rs",
            '#[test] fn device() { let _ = std::env::var("KILN_QUALIFICATION"); }\n',
        )
        self._write(
            "crates/kiln-memory/src/startup_environment.rs",
            'pub fn capture(name: &str) { let _ = std::env::var_os(name); }\n',
        )
        self._write(
            "crates/kiln-train/src/credential_provider.rs",
            'pub fn resolve(name: &str) { let _ = std::env::var(name); }\n',
        )
        contract = runtime_env.build_contract(self.root)
        self.assertEqual(
            self._entry(contract, "reads", "KILN_SERVER_PORT")["classification"],
            "public_stable",
        )
        self.assertEqual(
            self._entry(contract, "reads", "KILN_DEMO_ARCH")["classification"],
            "build_time",
        )
        self.assertEqual(
            self._entry(contract, "reads", "KILN_QUALIFICATION")["classification"],
            "test_only",
        )
        dynamic = {
            (entry["path"], entry["classification"])
            for entry in contract["reads"]
            if entry["argument"] == "name"
        }
        self.assertEqual(
            dynamic,
            {
                (
                    "crates/kiln-memory/src/startup_environment.rs",
                    "startup_safety",
                ),
                (
                    "crates/kiln-train/src/credential_provider.rs",
                    "credential_provider",
                ),
            },
        )

    def test_production_process_environment_mutation_is_forbidden(self) -> None:
        self._write_closed_runtime_boundaries()
        self._write(
            "crates/demo/src/lib.rs",
            'pub fn mutate() { unsafe { std::env::set_var("KILN_LIVE", "1"); } }\n',
        )
        with self.assertRaisesRegex(
            runtime_env.ScanError, "production process-environment mutation is forbidden"
        ):
            runtime_env.validate_policy(runtime_env.build_contract(self.root))

    def test_unclassified_production_runtime_read_is_forbidden(self) -> None:
        self._write_closed_runtime_boundaries()
        self._write(
            "crates/demo/src/lib.rs",
            'pub fn read() { let _ = std::env::var("KILN_LIVE"); }\n',
        )
        with self.assertRaisesRegex(
            runtime_env.ScanError,
            "direct production runtime environment reads are forbidden",
        ):
            runtime_env.validate_policy(runtime_env.build_contract(self.root))

    def test_closed_runtime_read_boundary_cannot_expand(self) -> None:
        self._write_closed_runtime_boundaries()
        self._write(
            "crates/kiln-memory/src/startup_environment.rs",
            '''
pub fn capture(name: &str, fallback: &str) {
    let _ = std::env::var_os(name);
    let _ = std::env::var_os(fallback);
}
''',
        )
        with self.assertRaisesRegex(
            runtime_env.ScanError,
            "startup-safety direct-read boundary must remain exactly one closed call shape",
        ):
            runtime_env.validate_policy(runtime_env.build_contract(self.root))

    def test_contract_check_rejects_new_accesses(self) -> None:
        self._write("crates/demo/src/lib.rs", "pub fn clean() {}\n")
        expected = runtime_env.build_contract(self.root)
        self._write(
            "crates/demo/src/lib.rs",
            'pub fn added() { let _ = std::env::var("KILN_NEW"); }\n',
        )
        actual = runtime_env.build_contract(self.root)
        errors = io.StringIO()
        with redirect_stderr(errors):
            matches = runtime_env.check_contract(expected, actual)
        self.assertFalse(matches)
        self.assertIn("new direct accesses", errors.getvalue())
        self.assertIn("KILN_NEW", errors.getvalue())

    def test_committed_contract_matches_source_tree(self) -> None:
        expected = json.loads(CONTRACT_PATH.read_text(encoding="utf-8"))
        actual = runtime_env.build_contract(ROOT)
        self.assertEqual(expected, actual)
        self.assertEqual(
            REPORT_PATH.read_text(encoding="utf-8"),
            runtime_env.render_report(actual),
        )


if __name__ == "__main__":
    unittest.main()
