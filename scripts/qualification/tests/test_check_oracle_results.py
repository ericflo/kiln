from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock


QUALIFICATION_DIR = Path(__file__).resolve().parents[1]
ROOT = Path(__file__).resolve().parents[3]
if str(QUALIFICATION_DIR) not in sys.path:
    sys.path.insert(0, str(QUALIFICATION_DIR))

import check_oracle_results as checker
import rocm_hf_layer_attribution as layer_attribution
import rocm_hf_next_token_oracle as hf_next_token
import rocm_hf_path_attribution as path_attribution


RETAINED_HF_RESULT = (
    ROOT
    / "qualification/oracle-results/rocm/strix-halo/"
    "20260719t003452-rocm-strix-halo-hf-next-token-first-divergence-v1.json"
)


class CheckOracleResultsTests(unittest.TestCase):
    def test_dispatches_each_known_schema_and_forwards_source_requirement(self) -> None:
        cases = (
            (hf_next_token.SCHEMA, hf_next_token),
            (layer_attribution.SCHEMA, layer_attribution),
            (path_attribution.SCHEMA, path_attribution),
        )
        with tempfile.TemporaryDirectory() as directory:
            for index, (schema, module) in enumerate(cases):
                path = Path(directory) / f"{index}.json"
                path.write_text(json.dumps({"schema": schema}), encoding="ascii")
                expected = {"schema": schema, "result_sha256": f"sha256:{index:064x}"}
                with mock.patch.object(module, "validate_result", return_value=expected) as run:
                    with mock.patch.dict(checker.VALIDATORS, {schema: module.validate_result}):
                        self.assertEqual(
                            checker.validate(path, require_current_source=True), expected
                        )
                run.assert_called_once_with(path, require_current_source=True)

    def test_rejects_unknown_schema_before_dispatch(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "unknown.json"
            path.write_text(json.dumps({"schema": "kiln.unknown.v1"}), encoding="ascii")
            with self.assertRaisesRegex(checker.OracleResultError, "unsupported"):
                checker.validate(path)

    def test_dispatcher_validates_current_retained_result(self) -> None:
        result = checker.validate(RETAINED_HF_RESULT)
        self.assertEqual(result["schema"], hf_next_token.SCHEMA)


if __name__ == "__main__":
    unittest.main()
