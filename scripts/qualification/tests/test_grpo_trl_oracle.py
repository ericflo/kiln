from __future__ import annotations

import hashlib
import importlib.util
import json
import math
import sys
import unittest
from pathlib import Path


MODULE_PATH = Path(__file__).resolve().parents[1] / "grpo_trl_oracle.py"
SPEC = importlib.util.spec_from_file_location("grpo_trl_oracle", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
grpo_trl_oracle = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = grpo_trl_oracle
SPEC.loader.exec_module(grpo_trl_oracle)


FIXTURE_PATH = (
    Path(__file__).resolve().parents[3]
    / "crates/kiln-train/tests/fixtures/grpo_trl_oracle_v1.json"
)


class GrpoTrlOracleContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.raw = FIXTURE_PATH.read_bytes()
        cls.fixture = json.loads(cls.raw)

    def test_fixture_is_canonical_and_source_pinned(self) -> None:
        self.assertEqual(
            self.raw,
            grpo_trl_oracle.canonical_bytes(self.fixture),
        )
        self.assertEqual(self.fixture["schema"], grpo_trl_oracle.SCHEMA)
        oracle = self.fixture["oracle"]
        self.assertEqual(oracle["trl_version"], grpo_trl_oracle.TRL_VERSION)
        self.assertEqual(oracle["trl_commit"], grpo_trl_oracle.TRL_COMMIT)
        self.assertEqual(
            oracle["trl_grpo_trainer_sha256"],
            "sha256:" + grpo_trl_oracle.TRL_GRPO_TRAINER_SHA256,
        )
        self.assertEqual(oracle["torch_version"], grpo_trl_oracle.TORCH_VERSION)
        self.assertEqual(oracle["torch_commit"], grpo_trl_oracle.TORCH_COMMIT)
        self.assertEqual(
            oracle["execution"],
            "TRL GRPOTrainer._compute_loss + PyTorch autograd/AdamW",
        )

    def test_fixture_inputs_bind_the_generator_cases(self) -> None:
        encoded = json.dumps(
            grpo_trl_oracle.CASES,
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
        digest = "sha256:" + hashlib.sha256(encoded).hexdigest()
        self.assertEqual(
            self.fixture["oracle"]["fixture_inputs_sha256"], digest
        )

        fixture_inputs = []
        for case in self.fixture["cases"]:
            fixture_inputs.append(
                {key: value for key, value in case.items() if key not in {"expected", "kl_estimator", "loss_normalizer"}}
            )
        self.assertEqual(fixture_inputs, grpo_trl_oracle.CASES)

    def test_cases_cover_distinct_policy_semantics(self) -> None:
        cases = {case["name"]: case for case in self.fixture["cases"]}
        self.assertEqual(len(cases), len(self.fixture["cases"]))
        self.assertEqual(
            set(cases),
            {
                "token_positive_asymmetric_k3",
                "token_negative_lower_clip_k3",
                "sequence_gspo_k3",
                "cispo_upper_weight_cap_k3",
                "no_importance_correction_keeps_k3",
            },
        )
        self.assertGreater(
            cases["token_positive_asymmetric_k3"]["expected"]["above_clip_count"],
            0,
        )
        self.assertGreater(
            cases["token_negative_lower_clip_k3"]["expected"]["below_clip_count"],
            0,
        )
        self.assertEqual(
            len(cases["sequence_gspo_k3"]["expected"]["observed_importance_ratios"]),
            1,
        )
        self.assertEqual(
            cases["cispo_upper_weight_cap_k3"]["expected"]["above_clip_count"],
            1,
        )
        self.assertEqual(
            cases["no_importance_correction_keeps_k3"]["expected"]["observed_importance_ratios"],
            [1.0, 1.0, 1.0, 1.0],
        )

    def test_every_numeric_observation_is_finite_and_shape_aligned(self) -> None:
        def assert_finite(value, path: str) -> None:
            if isinstance(value, bool) or value is None:
                return
            if isinstance(value, (int, float)):
                self.assertTrue(math.isfinite(value), path)
                return
            if isinstance(value, list):
                for index, item in enumerate(value):
                    assert_finite(item, f"{path}[{index}]")
                return
            if isinstance(value, dict):
                for key, item in value.items():
                    assert_finite(item, f"{path}.{key}")

        assert_finite(self.fixture, "fixture")
        for case in self.fixture["cases"]:
            token_count = len(case["policy_log_probs"])
            self.assertGreater(token_count, 0)
            self.assertEqual(len(case["behavior_log_probs"]), token_count)
            self.assertEqual(len(case["kl_reference_log_probs"]), token_count)
            expected = case["expected"]
            for key in (
                "policy_log_prob_grad",
                "token_importance_ratios",
                "k3_per_token",
                "adamw_parameter",
                "adamw_exp_avg",
                "adamw_exp_avg_sq",
            ):
                self.assertEqual(len(expected[key]), token_count, f"{case['name']} {key}")


if __name__ == "__main__":
    unittest.main()
