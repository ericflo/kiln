from __future__ import annotations

import copy
import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path


QUALIFICATION_DIR = Path(__file__).resolve().parents[1]
ROOT = Path(__file__).resolve().parents[3]
SPEC = importlib.util.spec_from_file_location(
    "qualification_workload", QUALIFICATION_DIR / "workload.py"
)
assert SPEC is not None and SPEC.loader is not None
workload_module = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = workload_module
SPEC.loader.exec_module(workload_module)


def valid_performance_workload() -> dict:
    return {
        "schema_version": 1,
        "workload_id": "performance-smoke-v1",
        "kind": "performance",
        "description": "Small deterministic performance contract fixture.",
        "determinism": {
            "seed": 7,
            "seed_delivery": "argv",
            "repetitions": 3,
            "case_order": "declared",
            "max_parallel_cases": 1,
            "network_access": "forbidden",
        },
        "variables": [],
        "variants": [
            {
                "id": "cpu-default",
                "description": "Default CPU fixture.",
                "backend": "cpu",
                "device_requirement": "none",
                "skip_policy": "allow",
                "cases": [
                    {
                        "id": "decode-throughput",
                        "description": "Measure deterministic decode throughput.",
                        "required": True,
                        "command": ["bench", "--seed", "${seed}"],
                        "working_directory": ".",
                        "environment": {},
                        "timeout_seconds": 60,
                        "expected_exit_codes": [0],
                        "output_assertions": [],
                        "result_protocol": {
                            "format": "qualification-case-result-v1",
                            "producer": "command",
                            "path_environment_variable": "KILN_QUALIFICATION_CASE_RESULT",
                            "declared_metrics": ["tokens_per_second"],
                        },
                    }
                ],
            }
        ],
        "comparison_policy": {
            "mode": "same_environment_performance",
            "variant_pairs": [],
            "backend_pairs": [],
            "allowed_effective_config_differences": [],
            "metric_rules": [
                {
                    "scope": "result",
                    "result_id": "decode-throughput",
                    "metric": "tokens_per_second",
                    "metric_class": "performance",
                    "operator": "not_less",
                    "absolute_tolerance": 0,
                    "relative_tolerance": 0.05,
                    "required": True,
                }
            ],
        },
    }


class WorkloadTests(unittest.TestCase):
    def test_checked_in_environment_and_correctness_workloads_validate(self) -> None:
        for name in ("environment-v1.json", "correctness-core-v1.json"):
            with self.subTest(name=name):
                path = ROOT / "qualification/workloads" / name
                workload = workload_module.load_workload(path)
                self.assertEqual(workload_module.validate_workload(workload), [])
                self.assertRegex(workload_module.workload_file_sha256(path), r"^sha256:[0-9a-f]{64}$")

    def test_valid_same_environment_performance_policy(self) -> None:
        self.assertEqual(workload_module.validate_workload(valid_performance_workload()), [])

    def test_unknown_and_missing_keys_are_rejected_at_nested_levels(self) -> None:
        value = valid_performance_workload()
        del value["description"]
        value["variants"][0]["cases"][0]["surprise"] = True
        errors = workload_module.validate_workload(value)
        self.assertTrue(any("missing keys: description" in error for error in errors))
        self.assertTrue(any("unknown keys: surprise" in error for error in errors))

    def test_duplicate_keys_and_non_finite_numbers_are_rejected_on_load(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            duplicate = Path(tmp) / "duplicate.json"
            duplicate.write_text('{"schema_version":1,"schema_version":1}')
            with self.assertRaises(workload_module.WorkloadLoadError):
                workload_module.load_workload(duplicate)

            non_finite = Path(tmp) / "nan.json"
            non_finite.write_text('{"value":NaN}')
            with self.assertRaises(workload_module.WorkloadLoadError):
                workload_module.load_workload(non_finite)

    def test_file_hash_uses_exact_validated_bytes(self) -> None:
        value = valid_performance_workload()
        with tempfile.TemporaryDirectory() as tmp:
            compact = Path(tmp) / "compact.json"
            pretty = Path(tmp) / "pretty.json"
            compact.write_text(json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n")
            pretty.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
            self.assertNotEqual(
                workload_module.workload_file_sha256(compact),
                workload_module.workload_file_sha256(pretty),
            )

    def test_file_hash_refuses_invalid_manifest(self) -> None:
        value = valid_performance_workload()
        value["surprise"] = True
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "invalid.json"
            path.write_text(json.dumps(value))
            with self.assertRaises(workload_module.WorkloadValidationError):
                workload_module.workload_file_sha256(path)

    def test_commands_must_be_argv_and_cannot_hide_a_shell_program(self) -> None:
        value = valid_performance_workload()
        value["variants"][0]["cases"][0]["command"] = "bench --seed 7"
        errors = workload_module.validate_workload(value)
        self.assertTrue(any("argv array" in error for error in errors))

        value = valid_performance_workload()
        value["variants"][0]["cases"][0]["command"] = [
            "bash",
            "-c",
            "bench --seed ${seed} && upload-results",
        ]
        errors = workload_module.validate_workload(value)
        self.assertTrue(any("shell command-string evaluation" in error for error in errors))
        self.assertTrue(any("placeholders must occupy" in error for error in errors))

    def test_placeholders_are_declared_and_seed_delivery_is_real(self) -> None:
        value = valid_performance_workload()
        value["variants"][0]["cases"][0]["command"][-1] = "${missing}"
        errors = workload_module.validate_workload(value)
        self.assertTrue(any("undeclared variable 'missing'" in error for error in errors))
        self.assertTrue(any("does not deliver determinism.seed" in error for error in errors))

        value = valid_performance_workload()
        value["determinism"]["seed_delivery"] = "fixed_fixture"
        errors = workload_module.validate_workload(value)
        self.assertTrue(any("limited to correctness" in error for error in errors))

    def test_required_accelerator_can_never_allow_a_skip(self) -> None:
        path = ROOT / "qualification/workloads/correctness-core-v1.json"
        value = workload_module.load_workload(path)
        value["variants"][0]["skip_policy"] = "allow"
        errors = workload_module.validate_workload(value)
        self.assertTrue(any("required device cannot allow" in error for error in errors))

    def test_runner_protocol_has_closed_known_metrics(self) -> None:
        value = valid_performance_workload()
        protocol = value["variants"][0]["cases"][0]["result_protocol"]
        protocol["producer"] = "runner"
        protocol["declared_metrics"] = ["tokens_per_second"]
        errors = workload_module.validate_workload(value)
        self.assertTrue(any("runner cannot produce metrics" in error for error in errors))
        self.assertTrue(any("must declare the case_pass metric" in error for error in errors))

    def test_declared_ab_policy_names_pair_and_exact_config_differences(self) -> None:
        value = valid_performance_workload()
        candidate = copy.deepcopy(value["variants"][0])
        candidate["id"] = "cpu-tuned"
        value["variants"].append(candidate)
        value["comparison_policy"] = {
            "mode": "declared_ab_variants",
            "variant_pairs": [
                {
                    "baseline_variant_id": "cpu-default",
                    "candidate_variant_id": "cpu-tuned",
                }
            ],
            "backend_pairs": [],
            "allowed_effective_config_differences": ["scheduler.max_batch_size"],
            "metric_rules": copy.deepcopy(value["comparison_policy"]["metric_rules"]),
        }
        self.assertEqual(workload_module.validate_workload(value), [])

        value["comparison_policy"]["allowed_effective_config_differences"] = []
        errors = workload_module.validate_workload(value)
        self.assertTrue(any("requires exact allowed effective-config" in error for error in errors))

    def test_cross_backend_policy_binds_variants_and_forbids_performance_metrics(self) -> None:
        value = workload_module.load_workload(
            ROOT / "qualification/workloads/correctness-core-v1.json"
        )
        rule = value["comparison_policy"]["metric_rules"][0]
        rule["metric_class"] = "performance"
        errors = workload_module.validate_workload(value)
        self.assertTrue(any("forbids performance metrics" in error for error in errors))

        value = workload_module.load_workload(
            ROOT / "qualification/workloads/correctness-core-v1.json"
        )
        value["comparison_policy"]["backend_pairs"][0]["variant_b_id"] = "rocm"
        errors = workload_module.validate_workload(value)
        self.assertTrue(any("backend does not match variant" in error for error in errors))

    def test_cross_backend_metrics_must_be_required_declared_results(self) -> None:
        value = workload_module.load_workload(
            ROOT / "qualification/workloads/correctness-core-v1.json"
        )
        value["variants"][1]["cases"][0]["required"] = False
        errors = workload_module.validate_workload(value)
        self.assertTrue(any("is not required by variant 'vulkan'" in error for error in errors))

        value = workload_module.load_workload(
            ROOT / "qualification/workloads/correctness-core-v1.json"
        )
        value["comparison_policy"]["metric_rules"][0]["metric"] = "max_abs_error"
        errors = workload_module.validate_workload(value)
        self.assertTrue(any("is not declared by result" in error for error in errors))

    def test_schema_files_are_closed_and_match_validator_contracts(self) -> None:
        schema = json.loads((ROOT / "qualification/schema/workload-v1.schema.json").read_text())
        self.assertFalse(schema["additionalProperties"])
        self.assertEqual(set(schema["required"]), workload_module.TOP_LEVEL_KEYS)
        self.assertEqual(set(schema["properties"]), workload_module.TOP_LEVEL_KEYS)
        contracts = [
            (schema["$defs"]["determinism"], workload_module.DETERMINISM_KEYS),
            (schema["$defs"]["variable"], workload_module.VARIABLE_KEYS),
            (
                schema["$defs"]["variable"]["properties"]["constraints"],
                workload_module.CONSTRAINT_KEYS,
            ),
            (schema["$defs"]["variant"], workload_module.VARIANT_KEYS),
            (schema["$defs"]["case"], workload_module.CASE_KEYS),
            (schema["$defs"]["outputAssertion"], workload_module.OUTPUT_ASSERTION_KEYS),
            (
                schema["$defs"]["case"]["properties"]["result_protocol"],
                workload_module.RESULT_PROTOCOL_KEYS,
            ),
            (schema["$defs"]["comparisonPolicy"], workload_module.COMPARISON_KEYS),
            (schema["$defs"]["variantPair"], workload_module.VARIANT_PAIR_KEYS),
            (schema["$defs"]["backendPair"], workload_module.BACKEND_PAIR_KEYS),
            (schema["$defs"]["metricRule"], workload_module.METRIC_RULE_KEYS),
        ]
        for contract, keys in contracts:
            with self.subTest(keys=keys):
                self.assertFalse(contract["additionalProperties"])
                self.assertEqual(set(contract["required"]), keys)
                self.assertEqual(set(contract["properties"]), keys)

        case_result = json.loads(
            (ROOT / "qualification/schema/case-result-v1.schema.json").read_text()
        )
        self.assertFalse(case_result["additionalProperties"])
        self.assertEqual(case_result["properties"]["schema_version"]["const"], 1)


if __name__ == "__main__":
    unittest.main()
