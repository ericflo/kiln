from __future__ import annotations

import importlib.util
import json
import math
import sys
import tempfile
import unittest
from pathlib import Path


QUALIFICATION_DIR = Path(__file__).resolve().parents[1]
ROOT = QUALIFICATION_DIR.parents[1]
sys.path.insert(0, str(QUALIFICATION_DIR))
SPEC = importlib.util.spec_from_file_location(
    "qualification_serve_rocm_graph_correctness",
    QUALIFICATION_DIR / "serve_rocm_graph_correctness.py",
)
assert SPEC is not None and SPEC.loader is not None
correctness = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = correctness
SPEC.loader.exec_module(correctness)


def response_fixture(*, second_token: int = 4, second_logprob: float = -0.2) -> dict:
    digest = "sha256:" + "a" * 64
    return {
        "choices": [
            {
                "finish_reason": "length",
                "message": {"content": "answer", "reasoning_content": None},
                "rollout_provenance": {
                    "schema": "kiln.rollout-provenance.v1",
                    "input_token_ids": [1, 2, 3, second_token],
                    "prompt_token_count": 2,
                    "prompt_messages_sha256": digest,
                    "scored_payload_sha256": digest,
                    "action_tokens": [
                        {
                            "sequence_index": 2,
                            "token_id": 3,
                            "source": "sampled",
                            "behavior_logprob": -0.1,
                        },
                        {
                            "sequence_index": 3,
                            "token_id": second_token,
                            "source": "sampled",
                            "behavior_logprob": second_logprob,
                        },
                    ],
                    "behavior_policy": {"base_model_sha256": digest},
                    "tokenizer": {"vocab_sha256": digest},
                    "sampling": {"temperature": 1.0, "top_k": 1},
                    "seed": 7,
                    "generation_backend": "rocm",
                },
            }
        ],
        "usage": {"prompt_tokens": 2, "completion_tokens": 2, "total_tokens": 4},
    }


class ServeRocmGraphCorrectnessTests(unittest.TestCase):
    def test_completion_request_rejects_invalid_bounds_before_http(self) -> None:
        with self.assertRaisesRegex(correctness.CorrectnessError, "max_tokens"):
            correctness.completion_request(1, "test", "prompt", 7, max_tokens=0)
        with self.assertRaisesRegex(correctness.CorrectnessError, "timeout_seconds"):
            correctness.completion_request(
                1, "test", "prompt", 7, timeout_seconds=math.inf
            )

    def test_parse_completion_preserves_exact_actions_and_logprobs(self) -> None:
        record = correctness.parse_completion(response_fixture(), "short", 7)
        self.assertEqual(record.scenario, "short")
        self.assertEqual(
            record.action_tokens,
            ((2, 3, "sampled", -0.1), (3, 4, "sampled", -0.2)),
        )
        self.assertEqual(record.sampled_logprobs, (-0.1, -0.2))
        self.assertEqual(record.semantic["usage"]["completion_tokens"], 2)

    def test_parse_completion_rejects_non_finite_positive_and_misaligned_actions(self) -> None:
        for invalid in (math.nan, math.inf, 0.01):
            with self.subTest(invalid=invalid):
                with self.assertRaisesRegex(correctness.CorrectnessError, "log-probability"):
                    correctness.parse_completion(
                        response_fixture(second_logprob=invalid), "short", 7
                    )
        malformed = response_fixture()
        malformed["choices"][0]["rollout_provenance"]["action_tokens"][1][
            "token_id"
        ] = 9
        with self.assertRaisesRegex(correctness.CorrectnessError, "input_token_ids"):
            correctness.parse_completion(malformed, "short", 7)

    def test_mismatch_counts_separate_trace_token_and_logprob_drift(self) -> None:
        baseline = correctness.parse_completion(response_fixture(), "short", 7)
        token_drift = correctness.parse_completion(
            response_fixture(second_token=5), "short", 7
        )
        logprob_drift = correctness.parse_completion(
            response_fixture(second_logprob=-0.3), "short", 7
        )
        self.assertEqual(
            correctness.mismatch_counts((baseline,), (token_drift,)),
            {
                "output_mismatch_count": 1,
                "token_id_mismatch_count": 1,
                "behavior_logprob_mismatch_count": 0,
            },
        )
        self.assertEqual(
            correctness.mismatch_counts((baseline,), (logprob_drift,)),
            {
                "output_mismatch_count": 1,
                "token_id_mismatch_count": 0,
                "behavior_logprob_mismatch_count": 1,
            },
        )
        self.assertEqual(
            correctness.first_mismatch((baseline,), (token_drift,)),
            {
                "action_index": 1,
                "left_action": [3, 4, "sampled", -0.2],
                "right_action": [3, 5, "sampled", -0.2],
                "scenario": "short",
            },
        )
        self.assertIsNone(correctness.first_mismatch((baseline,), (baseline,)))

    def test_checked_in_workload_exactly_matches_driver_contract(self) -> None:
        workload = json.loads(
            (
                ROOT
                / "qualification/workloads/serving-rocm-graph-correctness-v1.json"
            ).read_text()
        )
        self.assertEqual(workload["kind"], "correctness")
        self.assertEqual(
            workload["comparison_policy"],
            {
                "mode": "self_contained_correctness",
                "variant_pairs": [],
                "backend_pairs": [],
                "metric_rules": [],
            },
        )
        variant = workload["variants"][0]
        self.assertEqual(variant["id"], correctness.VARIANT_ID)
        self.assertEqual(variant["effective_config"], correctness.EFFECTIVE_CONFIG)
        case = variant["cases"][0]
        self.assertEqual(case["id"], correctness.CASE_ID)
        self.assertEqual(
            case["result_protocol"]["declared_metrics"],
            sorted(correctness.METRIC_DEFINITIONS),
        )

    def test_metric_contract_is_closed_finite_and_sorted(self) -> None:
        values = {name: 0 for name in correctness.METRIC_DEFINITIONS}
        metrics = correctness.metrics_from_values(values)
        self.assertEqual(
            [metric["name"] for metric in metrics],
            sorted(correctness.METRIC_DEFINITIONS),
        )
        missing = dict(values)
        missing.pop("request_failure_count")
        with self.assertRaisesRegex(correctness.CorrectnessError, "metric set mismatch"):
            correctness.metrics_from_values(missing)
        for invalid in (-1, math.inf, math.nan, True):
            malformed = dict(values)
            malformed["request_failure_count"] = invalid
            with self.assertRaises(correctness.CorrectnessError):
                correctness.metrics_from_values(malformed)

    def test_result_status_is_explicitly_bound_to_pass_details(self) -> None:
        metrics = correctness.metrics_from_values(
            {name: 0 for name in correctness.METRIC_DEFINITIONS}
        )
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "result.json"
            correctness.write_result(path, metrics, '{"trace":"sha256"}', 1.0)
            self.assertEqual(json.loads(path.read_text())["status"], "passed")
            correctness.write_result(path, metrics, "mismatch", 1.0)
            self.assertEqual(json.loads(path.read_text())["status"], "failed")


if __name__ == "__main__":
    unittest.main()
