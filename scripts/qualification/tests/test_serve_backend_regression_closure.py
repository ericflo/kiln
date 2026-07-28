from __future__ import annotations

import importlib.util
import json
import sys
import unittest
from pathlib import Path


QUALIFICATION_DIR = Path(__file__).resolve().parents[1]
ROOT = QUALIFICATION_DIR.parents[1]
sys.path.insert(0, str(QUALIFICATION_DIR))
SPEC = importlib.util.spec_from_file_location(
    "qualification_serve_backend_regression_closure",
    QUALIFICATION_DIR / "serve_backend_regression_closure.py",
)
assert SPEC is not None and SPEC.loader is not None
closure = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = closure
SPEC.loader.exec_module(closure)


def stream_result(
    *,
    name: str = "kv-growth-pressure",
    prompt_tokens: int = 274,
    completion_tokens: int = closure.PRESSURE_MAX_TOKENS,
    content: str = "000000 ",
) -> closure.mixed.StreamResult:
    return closure.mixed.StreamResult(
        name=name,
        marker="marker",
        started=1.0,
        finished=2.0,
        semantic_times=[1.25],
        token_ready_times=[1.25] * completion_tokens,
        token_queue_delays_ms=[0.0] * completion_tokens,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        usage_records=1,
        finish_reason="length",
        done=True,
        cancelled=False,
        error=None,
        token_ids=list(range(completion_tokens)),
        semantic_deltas=[
            {"choices": [{"index": 0, "delta": {"content": content}}]}
        ],
    )


class ServeBackendRegressionClosureTests(unittest.TestCase):
    def test_checked_in_workload_exactly_matches_driver_contract(self) -> None:
        manifest = json.loads(
            (
                ROOT
                / "qualification/workloads/serving-backend-regression-closure-v1.json"
            ).read_text()
        )
        variants = {variant["id"]: variant for variant in manifest["variants"]}
        self.assertEqual(set(variants), set(closure.VARIANTS))
        for variant_id in closure.VARIANTS:
            with self.subTest(variant=variant_id):
                variant = variants[variant_id]
                self.assertEqual(
                    variant["effective_config"],
                    closure.EFFECTIVE_CONFIGS[variant_id],
                )
                self.assertEqual(
                    variant["cases"][0]["id"], closure.CASE_IDS[variant_id]
                )
                self.assertEqual(
                    variant["cases"][0]["result_protocol"]["declared_metrics"],
                    sorted(closure.METRIC_DEFINITIONS),
                )

    def test_pressure_fixture_crosses_four_decode_block_boundaries(self) -> None:
        result = stream_result()
        growth_blocks = (
            (
                result.prompt_tokens
                + result.completion_tokens
                + closure.KV_BLOCK_SIZE
                - 1
            )
            // closure.KV_BLOCK_SIZE
            - (
                result.prompt_tokens + closure.KV_BLOCK_SIZE - 1
            )
            // closure.KV_BLOCK_SIZE
        )
        self.assertEqual(growth_blocks, 4)
        self.assertEqual(
            closure.stream_failures(result, closure.PRESSURE_MAX_TOKENS), []
        )

    def test_stream_gate_rejects_truncated_or_non_exact_output(self) -> None:
        truncated = stream_result(completion_tokens=8)
        failures = closure.stream_failures(
            truncated, closure.PRESSURE_MAX_TOKENS
        )
        self.assertTrue(any("expected 256" in failure for failure in failures))
        malformed = stream_result(content="wrong")
        failures = closure.stream_failures(
            malformed, closure.PRESSURE_MAX_TOKENS
        )
        self.assertTrue(any("response oracle failed" in failure for failure in failures))

    def test_structured_reclaim_requires_positive_block_evidence(self) -> None:
        event = closure.mixed.ObservedEvent(
            observed=1.0,
            category="prefix_cache_reclaim",
            message="reclaimed",
            fields={"reclaimed_prefix_blocks": 7, "requested_blocks": 1},
        )
        self.assertEqual(closure.reclaimed_block_count([event]), 7)
        invalid = closure.mixed.ObservedEvent(
            observed=1.0,
            category="prefix_cache_reclaim",
            message="reclaimed",
            fields={"reclaimed_prefix_blocks": 0},
        )
        with self.assertRaisesRegex(closure.ClosureError, "positive"):
            closure.reclaimed_block_count([invalid])

    def test_shared_log_classifier_names_prefix_reclamation(self) -> None:
        self.assertEqual(
            closure.mixed.classify_server_event(
                "reclaimed unleased prefix-cache blocks for live decode growth",
                {"reclaimed_prefix_blocks": 7, "requested_blocks": 1},
            ),
            "prefix_cache_reclaim",
        )


if __name__ == "__main__":
    unittest.main()
