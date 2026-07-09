from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


QUALIFICATION_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(QUALIFICATION_DIR))
SPEC = importlib.util.spec_from_file_location("qualification_receipt", QUALIFICATION_DIR / "receipt.py")
assert SPEC is not None and SPEC.loader is not None
receipt_module = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = receipt_module
SPEC.loader.exec_module(receipt_module)

HASH = "sha256:" + "a" * 64
COMMIT = "b" * 40


def valid_receipt() -> dict:
    return {
        "schema_version": 1,
        "receipt_id": "rocm-strix-environment-20260709",
        "created_at_utc": "2026-07-09T20:00:02Z",
        "source": {
            "tree_hash_format": "kiln-source-tree-v1",
            "tree_hash": HASH,
            "git_commit": COMMIT,
            "git_worktree_clean": True,
        },
        "qualification": {
            "kind": "environment",
            "backend": "rocm",
            "profile": "environment-v1",
            "verdict": "passed",
            "started_at_utc": "2026-07-09T20:00:00Z",
            "finished_at_utc": "2026-07-09T20:00:02Z",
            "duration_seconds": 2.0,
            "command": ["python3", "scripts/qualification/environment.py", "--backend", "rocm"],
        },
        "environment": {
            "host_id": "strix-halo",
            "os": {
                "name": "Arch Linux",
                "version": "rolling",
                "kernel": "6.15.4",
                "architecture": "x86_64",
            },
            "device": {
                "name": "AMD Radeon 8060S",
                "architecture": "gfx1151",
                "memory_bytes": 103079215104,
                "unified_memory": True,
                "driver": "amdgpu 6.15.4",
            },
            "runtime": {"rocm": "7.2.4", "hip": "7.2.4"},
            "compiler": {"rustc": "1.90.0", "hipcc": "7.2.4"},
        },
        "model": None,
        "workload": None,
        "effective_config": {},
        "results": [
            {
                "id": "device-detected",
                "required": True,
                "status": "passed",
                "duration_seconds": 0.1,
                "metrics": [],
                "details": None,
            }
        ],
        "metrics": [],
        "artifacts": [],
        "unsupported": [],
        "notes": [],
    }


class ReceiptTests(unittest.TestCase):
    def test_valid_environment_receipt(self) -> None:
        self.assertEqual(receipt_module.validate_receipt(valid_receipt()), [])

    def test_unknown_and_missing_keys_are_rejected(self) -> None:
        value = valid_receipt()
        del value["notes"]
        value["surprise"] = True
        errors = receipt_module.validate_receipt(value)
        self.assertTrue(any("missing keys: notes" in error for error in errors))
        self.assertTrue(any("unknown keys: surprise" in error for error in errors))

    def test_passed_receipt_must_be_clean_and_cannot_skip_required_test(self) -> None:
        value = valid_receipt()
        value["source"]["git_worktree_clean"] = False
        value["results"][0]["status"] = "skipped"
        errors = receipt_module.validate_receipt(value)
        self.assertTrue(any("git_worktree_clean=true" in error for error in errors))
        self.assertTrue(any("skipped required" in error for error in errors))

    def test_failed_receipt_requires_a_required_failure(self) -> None:
        value = valid_receipt()
        value["qualification"]["verdict"] = "failed"
        errors = receipt_module.validate_receipt(value)
        self.assertTrue(any("failed receipt must contain" in error for error in errors))

    def test_result_details_have_a_strict_size_limit(self) -> None:
        value = valid_receipt()
        value["results"][0]["details"] = (
            "x" * (receipt_module.MAX_RESULT_DETAIL_CHARACTERS + 1)
        )
        errors = receipt_module.validate_receipt(value)
        self.assertTrue(any("details must be at most" in error for error in errors))

    def test_duplicate_result_and_metric_names_are_rejected(self) -> None:
        value = valid_receipt()
        value["results"].append(copy.deepcopy(value["results"][0]))
        metric = {
            "name": "itl_p99_ms",
            "value": 12.0,
            "unit": "ms",
            "aggregation": "p99",
            "lower_is_better": True,
        }
        value["metrics"] = [metric, copy.deepcopy(metric)]
        errors = receipt_module.validate_receipt(value)
        self.assertTrue(any("duplicate id" in error for error in errors))
        self.assertTrue(any("duplicate metric" in error for error in errors))

    def test_timestamp_duration_must_be_consistent(self) -> None:
        value = valid_receipt()
        value["qualification"]["duration_seconds"] = 9.0
        errors = receipt_module.validate_receipt(value)
        self.assertTrue(any("differs from timestamps" in error for error in errors))

    def test_serving_receipt_requires_model_and_workload(self) -> None:
        value = valid_receipt()
        value["qualification"]["kind"] = "serving"
        errors = receipt_module.validate_receipt(value)
        self.assertTrue(any("model is required" in error for error in errors))
        self.assertTrue(any("workload is required" in error for error in errors))

    def test_correctness_receipt_requires_workload_but_not_model(self) -> None:
        value = valid_receipt()
        value["qualification"]["kind"] = "correctness"
        errors = receipt_module.validate_receipt(value)
        self.assertTrue(any("workload is required" in error for error in errors))
        self.assertFalse(any("model is required" in error for error in errors))

    def test_duplicate_json_key_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "duplicate.json"
            path.write_text('{"schema_version":1,"schema_version":1}')
            with self.assertRaises(receipt_module.ReceiptLoadError):
                receipt_module.load_receipt(path)

    def test_non_finite_json_number_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "nan.json"
            path.write_text('{"value":NaN}')
            with self.assertRaises(receipt_module.ReceiptLoadError):
                receipt_module.load_receipt(path)

    def test_receipt_bytes_require_plain_utf8_and_wrap_decode_errors(self) -> None:
        payloads = {
            "invalid-utf8": b"\xff{\"schema_version\":1}",
            "utf8-bom": b"\xef\xbb\xbf{\"schema_version\":1}",
            "utf16": '{"schema_version":1}'.encode("utf-16"),
        }
        with tempfile.TemporaryDirectory() as tmp:
            for name, payload in payloads.items():
                with self.subTest(name=name):
                    path = Path(tmp) / f"{name}.json"
                    path.write_bytes(payload)
                    with self.assertRaisesRegex(
                        receipt_module.ReceiptLoadError, "cannot load"
                    ):
                        receipt_module.load_receipt(path)

    def test_lossy_and_oversized_json_numbers_are_rejected(self) -> None:
        payloads = {
            "overflow": '{"value":1e400}',
            "underflow": '{"value":1e-4000}',
            "rounded-boundary": '{"value":8.9999999999999999}',
            "rounded-integer": '{"value":9007199254740993.0}',
            "rounded-subnormal": '{"value":4e-324}',
            "huge-integer": '{"value":1' + "0" * 5000 + "}",
        }
        with tempfile.TemporaryDirectory() as tmp:
            for name, payload in payloads.items():
                with self.subTest(name=name):
                    path = Path(tmp) / f"{name}.json"
                    path.write_text(payload)
                    with self.assertRaises(receipt_module.ReceiptLoadError):
                        receipt_module.load_receipt(path)

    def test_effective_config_has_canonical_finite_scalar_leaves(self) -> None:
        invalid_configs = [
            {"BadKey": True},
            {"scheduler": [1, 2]},
            {"scheduler": {"limit": 10**500}},
        ]
        for config in invalid_configs:
            with self.subTest(config=config):
                value = valid_receipt()
                value["effective_config"] = config
                errors = receipt_module.validate_receipt(value)
                self.assertTrue(any("effective_config" in error for error in errors))

    def test_current_source_mismatch_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            subprocess.run(["git", "init", "-q"], cwd=root, check=True)
            (root / "Cargo.toml").write_text("[workspace]\n")
            subprocess.run(["git", "add", "Cargo.toml"], cwd=root, check=True)
            errors = receipt_module.validate_receipt(
                valid_receipt(), root=root, require_current_source=True
            )
            self.assertTrue(any("current source tree" in error for error in errors))

    def test_local_artifact_is_rehashed(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            artifact = root / ".qualification" / "raw.log"
            artifact.parent.mkdir()
            artifact.write_bytes(b"hello\n")
            value = valid_receipt()
            value["artifacts"] = [
                {
                    "kind": "raw_log",
                    "location": "local_ignored",
                    "path": ".qualification/raw.log",
                    "sha256": "sha256:" + hashlib.sha256(b"hello\n").hexdigest(),
                    "bytes": 6,
                }
            ]
            self.assertEqual(
                receipt_module.validate_receipt(
                    value, root=root, require_local_artifacts=True
                ),
                [],
            )
            artifact.write_bytes(b"changed\n")
            errors = receipt_module.validate_receipt(
                value, root=root, require_local_artifacts=True
            )
            self.assertTrue(any("does not match local artifact" in error for error in errors))

    def test_schema_file_is_well_formed_and_closed(self) -> None:
        schema_path = Path(__file__).resolve().parents[3] / "qualification/schema/receipt-v1.schema.json"
        schema = json.loads(schema_path.read_text())
        self.assertEqual(schema["$schema"], "https://json-schema.org/draft/2020-12/schema")
        self.assertFalse(schema["additionalProperties"])
        self.assertEqual(schema["properties"]["schema_version"]["const"], 1)
        self.assertEqual(schema["properties"]["effective_config"]["$ref"], "#/$defs/configObject")
        self.assertEqual(len(schema["allOf"]), 2)
        self.assertEqual(
            schema["$defs"]["result"]["properties"]["details"]["oneOf"][0][
                "maxLength"
            ],
            receipt_module.MAX_RESULT_DETAIL_CHARACTERS,
        )
        self.assertEqual(
            schema["allOf"][0]["then"]["properties"]["workload"]["$ref"],
            "#/$defs/workload",
        )
        self.assertEqual(
            schema["allOf"][1]["then"]["properties"]["model"]["$ref"],
            "#/$defs/model",
        )

    def test_atomic_write_json_replaces_complete_document(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "nested" / "receipt.json"
            receipt_module.atomic_write_json(path, {"version": 1})
            self.assertEqual(json.loads(path.read_text()), {"version": 1})
            receipt_module.atomic_write_json(path, {"version": 2, "complete": True})
            self.assertEqual(
                json.loads(path.read_text()),
                {"version": 2, "complete": True},
            )
            self.assertEqual(list(path.parent.glob("*.tmp")), [])

    def test_python_validator_keys_and_enums_match_schema(self) -> None:
        schema_path = Path(__file__).resolve().parents[3] / "qualification/schema/receipt-v1.schema.json"
        schema = json.loads(schema_path.read_text())
        self.assertEqual(set(schema["required"]), receipt_module.TOP_LEVEL_KEYS)
        self.assertEqual(set(schema["properties"]), receipt_module.TOP_LEVEL_KEYS)

        object_contracts = [
            (schema["properties"]["source"], receipt_module.SOURCE_KEYS),
            (schema["properties"]["qualification"], receipt_module.QUALIFICATION_KEYS),
            (schema["properties"]["environment"], receipt_module.ENVIRONMENT_KEYS),
            (schema["properties"]["environment"]["properties"]["os"], receipt_module.OS_KEYS),
            (schema["properties"]["environment"]["properties"]["device"], receipt_module.DEVICE_KEYS),
            (schema["$defs"]["model"], receipt_module.MODEL_KEYS),
            (schema["$defs"]["weightFile"], receipt_module.WEIGHT_KEYS),
            (schema["$defs"]["workload"], receipt_module.WORKLOAD_KEYS),
            (schema["$defs"]["result"], receipt_module.RESULT_KEYS),
            (schema["$defs"]["metric"], receipt_module.METRIC_KEYS),
            (schema["$defs"]["artifact"], receipt_module.ARTIFACT_KEYS),
        ]
        for contract, expected_keys in object_contracts:
            with self.subTest(expected_keys=expected_keys):
                self.assertFalse(contract["additionalProperties"])
                self.assertEqual(set(contract["required"]), expected_keys)
                self.assertEqual(set(contract["properties"]), expected_keys)

        qualification = schema["properties"]["qualification"]["properties"]
        self.assertEqual(set(qualification["kind"]["enum"]), receipt_module.KINDS)
        self.assertEqual(set(qualification["backend"]["enum"]), receipt_module.BACKENDS)
        self.assertEqual(set(qualification["verdict"]["enum"]), receipt_module.VERDICTS)
        self.assertEqual(
            set(schema["$defs"]["result"]["properties"]["status"]["enum"]),
            receipt_module.RESULT_STATUSES,
        )
        self.assertEqual(
            set(schema["$defs"]["artifact"]["properties"]["location"]["enum"]),
            receipt_module.ARTIFACT_LOCATIONS,
        )


if __name__ == "__main__":
    unittest.main()
