from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path


QUALIFICATION_DIR = Path(__file__).resolve().parents[1]
ROOT = QUALIFICATION_DIR.parents[1]
sys.path.insert(0, str(QUALIFICATION_DIR))
SPEC = importlib.util.spec_from_file_location(
    "qualification_serve_rocm_graph_failure_containment",
    QUALIFICATION_DIR / "serve_rocm_graph_failure_containment.py",
)
assert SPEC is not None and SPEC.loader is not None
failure = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = failure
SPEC.loader.exec_module(failure)


class ServeRocmGraphFailureContainmentTests(unittest.TestCase):
    def test_checked_in_workload_exactly_matches_driver_contract(self) -> None:
        manifest = json.loads(
            (
                ROOT
                / "qualification/workloads/serving-rocm-graph-failure-containment-v1.json"
            ).read_text()
        )
        variant = manifest["variants"][0]
        self.assertEqual(variant["id"], failure.VARIANT_ID)
        self.assertEqual(variant["effective_config"], failure.EFFECTIVE_CONFIG)
        self.assertEqual(variant["device_requirement"], "required")
        self.assertEqual(variant["skip_policy"], "fail")
        case = variant["cases"][0]
        self.assertEqual(case["id"], failure.CASE_ID)
        self.assertEqual(
            case["result_protocol"]["declared_metrics"],
            sorted(failure.METRIC_DEFINITIONS),
        )

    def test_command_is_bounded_offline_and_runs_only_ignored_graph_corpus(self) -> None:
        command = failure.command()
        self.assertEqual(command[0], str(ROOT / "scripts/cargo-bounded.sh"))
        self.assertEqual(command[1], "test")
        self.assertIn("--locked", command)
        self.assertIn("--offline", command)
        self.assertIn("--no-default-features", command)
        self.assertIn("rocm_graph::tests::", command)
        self.assertEqual(command[-3:], ["--ignored", "--nocapture", "--test-threads=1"])

    def test_environment_requires_qualification_and_preserves_memory_floor(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            cargo = Path(tmp) / "cargo"
            cargo.write_text("#!/bin/sh\n")
            cargo.chmod(0o755)
            environment = failure.test_environment(
                {"CARGO": str(cargo), "HOME": tmp, "PATH": "/usr/bin"}
            )
        self.assertEqual(environment["KILN_QUALIFICATION"], "1")
        self.assertEqual(environment["KILN_CARGO_MIN_AVAILABLE_GIB"], "15")
        self.assertEqual(environment["KILN_CARGO_JOBS"], "1")
        self.assertEqual(environment["KILN_CARGO_CPU_QUOTA_PERCENT"], "50")
        self.assertEqual(
            environment["KILN_CARGO_SERVICE_RUNTIME_MAX_SECONDS"], "1140"
        )
        self.assertEqual(environment["KILN_ROCM_ARCHS"], "gfx1151")

    def test_output_parser_requires_exact_passing_names(self) -> None:
        output = "\n".join(
            f"test rocm_graph::tests::{name} ... ok" for name in failure.TESTS
        )
        self.assertEqual(failure.parse_passed_tests(output), set(failure.TESTS))
        self.assertEqual(failure.parse_passed_tests(output.replace(" ... ok", " ... FAILED")), set())

    def test_metrics_distinguish_each_containment_claim(self) -> None:
        values = {
            metric["name"]: metric["value"]
            for metric in failure.metrics(set(failure.TESTS))
        }
        self.assertEqual(values["passed_test_count"], 3)
        self.assertEqual(values["failed_test_count"], 0)
        self.assertEqual(values["shape_dependent_fallback_test_count"], 1)
        self.assertEqual(values["graph_lifecycle_parity_test_count"], 1)
        self.assertEqual(values["stale_generation_containment_test_count"], 1)


if __name__ == "__main__":
    unittest.main()
