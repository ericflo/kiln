from __future__ import annotations

import hashlib
import importlib.util
import json
import math
import sys
import unittest
from pathlib import Path


MODULE_PATH = Path(__file__).resolve().parents[1] / "adamw_pytorch_oracle.py"
SPEC = importlib.util.spec_from_file_location("adamw_pytorch_oracle", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
oracle = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = oracle
SPEC.loader.exec_module(oracle)

FIXTURE_PATH = (
    Path(__file__).resolve().parents[3]
    / "crates/kiln-optim/tests/fixtures/adamw_pytorch_oracle_v1.json"
)


class AdamwPytorchOracleContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.raw = FIXTURE_PATH.read_bytes()
        cls.fixture = json.loads(cls.raw)

    def test_fixture_is_canonical_and_source_pinned(self) -> None:
        self.assertEqual(self.raw, oracle.canonical_bytes(self.fixture))
        self.assertEqual(self.fixture["schema"], oracle.SCHEMA)
        source = self.fixture["oracle"]
        self.assertEqual(source["seed"], oracle.SEED)
        self.assertEqual(source["torch_version"], oracle.TORCH_VERSION)
        self.assertEqual(source["torch_commit"], oracle.TORCH_COMMIT)
        self.assertEqual(
            source["torch_adamw_sha256"],
            "sha256:" + oracle.TORCH_ADAMW_SHA256,
        )
        self.assertEqual(
            source["torch_adam_sha256"],
            "sha256:" + oracle.TORCH_ADAM_SHA256,
        )
        self.assertEqual(source["implementation"], "torch.optim.AdamW")
        self.assertFalse(source["foreach"])
        self.assertFalse(source["fused"])
        self.assertFalse(source["capturable"])
        self.assertFalse(source["differentiable"])
        tolerances = self.fixture["tolerances"]
        self.assertEqual(tolerances["bfloat16_parameter_max_ulp"], 1)
        self.assertEqual(tolerances["bfloat16_first_moment_max_ulp"], 4)
        self.assertEqual(tolerances["bfloat16_second_moment_max_ulp"], 3)
        self.assertIn("rounds each output once", tolerances["bfloat16_reason"])

    def test_fixture_inputs_bind_all_generator_cases(self) -> None:
        payload = json.dumps(
            oracle.CASES,
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
        self.assertEqual(
            self.fixture["oracle"]["fixture_inputs_sha256"],
            "sha256:" + hashlib.sha256(payload).hexdigest(),
        )
        observed = []
        source_cases = {case["id"]: case for case in oracle.CASES}
        for case in self.fixture["cases"]:
            source = source_cases[case["id"]]
            observed.append(
                {
                    "id": case["id"],
                    "class": case["class"],
                    "dtype": case["dtype"],
                    "initial_parameter": source["initial_parameter"],
                    "gradients": source["gradients"],
                    "hyperparameters": case["hyperparameters"],
                }
            )
        self.assertEqual(observed, oracle.CASES)

    def test_cases_cover_low_and_ordinary_f32_and_bf16(self) -> None:
        cases = {case["id"]: case for case in self.fixture["cases"]}
        self.assertEqual(
            set(cases),
            {
                "ordinary_float32",
                "ordinary_bfloat16",
                "low_gradient_float32",
                "low_gradient_bfloat16",
            },
        )
        for case in cases.values():
            self.assertEqual(len(case["stored_initial_parameter"]), 8)
            self.assertEqual(len(case["stored_gradients"]), oracle.STEP_COUNT)
            self.assertEqual(len(case["trajectory"]), oracle.STEP_COUNT)
            self.assertEqual(
                [step["step"] for step in case["trajectory"]],
                list(range(1, oracle.STEP_COUNT + 1)),
            )

    def test_precision_contract_has_no_hidden_master(self) -> None:
        for case in self.fixture["cases"]:
            contract = case["state_contract"]
            self.assertEqual(contract["parameter_dtype"], case["dtype"])
            self.assertEqual(contract["first_moment_dtype"], case["dtype"])
            self.assertEqual(contract["second_moment_dtype"], case["dtype"])
            self.assertEqual(contract["step_dtype"], "float32")
            self.assertFalse(contract["separate_master_parameter"])

    def test_low_gradient_cases_straddle_epsilon_and_bf16_update_floor(self) -> None:
        cases = {case["id"]: case for case in self.fixture["cases"]}
        for dtype in ("float32", "bfloat16"):
            case = cases[f"low_gradient_{dtype}"]
            magnitudes = [abs(value) for row in case["stored_gradients"] for value in row]
            self.assertLess(min(magnitudes), case["hyperparameters"]["eps"] * 1e-3)
            self.assertGreater(max(magnitudes), case["hyperparameters"]["eps"] * 100)
        bf16 = cases["low_gradient_bfloat16"]
        first = bf16["trajectory"][0]
        self.assertEqual(first["parameter"][:6], bf16["stored_initial_parameter"][:6])
        self.assertTrue(any(value != 0 for value in first["exp_avg"][:6]))
        self.assertTrue(any(value != 0 for value in first["exp_avg_sq"][:6]))

    def test_all_observations_are_finite_and_shape_aligned(self) -> None:
        for case in self.fixture["cases"]:
            width = len(case["stored_initial_parameter"])
            for row in case["stored_gradients"]:
                self.assertEqual(len(row), width)
                self.assertTrue(all(math.isfinite(value) for value in row))
            for step in case["trajectory"]:
                for field in ("parameter", "exp_avg", "exp_avg_sq"):
                    self.assertEqual(len(step[field]), width)
                    self.assertTrue(all(math.isfinite(value) for value in step[field]))


if __name__ == "__main__":
    unittest.main()
