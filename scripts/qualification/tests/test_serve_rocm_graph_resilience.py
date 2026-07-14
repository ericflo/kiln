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
    "qualification_serve_rocm_graph_resilience",
    QUALIFICATION_DIR / "serve_rocm_graph_resilience.py",
)
assert SPEC is not None and SPEC.loader is not None
resilience = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = resilience
SPEC.loader.exec_module(resilience)


def stream_result(name: str, content: str = "token") -> resilience.mixed.StreamResult:
    return resilience.mixed.StreamResult(
        name=name,
        marker="marker",
        started=1.0,
        finished=2.0,
        semantic_times=[1.25],
        token_ready_times=[1.25, 1.5],
        token_queue_delays_ms=[0.0, 0.0],
        prompt_tokens=4,
        completion_tokens=resilience.MAX_TOKENS,
        usage_records=1,
        finish_reason="length",
        done=True,
        cancelled=False,
        error=None,
        semantic_deltas=[
            {
                "id": "dynamic-request-id",
                "created": 1,
                "choices": [{"index": 0, "delta": {"content": content}}],
            }
        ],
    )


def graph(**overrides: int) -> dict[str, int]:
    value = {
        "budget_evictions": 0,
        "byte_budget_rejections": 0,
        "capture_successes": 4,
        "failures": 0,
        "fallback_graph_cache_byte_budget": 0,
        "peak_retained_bytes": 1024,
        "pre_capture_byte_budget_skips": 0,
        "replay_successes": 8,
    }
    value.update(overrides)
    return value


class ServeRocmGraphResilienceTests(unittest.TestCase):
    def test_checked_in_workload_exactly_matches_driver_contract(self) -> None:
        manifest = json.loads(
            (
                ROOT
                / "qualification/workloads/serving-rocm-graph-resilience-v1.json"
            ).read_text()
        )
        self.assertEqual(manifest["workload_id"], "serving-rocm-graph-resilience-v1")
        self.assertEqual(len(manifest["variants"]), 1)
        variant = manifest["variants"][0]
        self.assertEqual(variant["id"], resilience.VARIANT_ID)
        self.assertEqual(variant["effective_config"], resilience.EFFECTIVE_CONFIG)
        self.assertEqual(variant["device_requirement"], "required")
        self.assertEqual(variant["skip_policy"], "fail")
        case = variant["cases"][0]
        self.assertEqual(case["id"], resilience.CASE_ID)
        self.assertEqual(
            case["result_protocol"]["declared_metrics"],
            sorted(resilience.METRIC_DEFINITIONS),
        )
        self.assertEqual(
            case["command"],
            [
                "python3",
                "scripts/qualification/serve_rocm_graph_resilience.py",
                "--model-path",
                "${model_path}",
                "--seed",
                "${seed}",
            ],
        )

    def test_effective_config_closes_budget_and_concurrency_matrix(self) -> None:
        workload = resilience.EFFECTIVE_CONFIG["workload"]
        runtime = resilience.EFFECTIVE_CONFIG["runtime"]
        self.assertEqual(
            tuple(workload["concurrency_levels"].values()),
            (1, 8, 16, 32, 64),
        )
        self.assertEqual(workload["request_count_per_arm"], 121)
        self.assertEqual(runtime["rocm_graph_cache_entries"], 64)
        self.assertEqual(
            runtime["rocm_graph_cache_max_bytes_by_arm"],
            {"headroom": 1 << 30, "tight": 64 << 20},
        )
        self.assertEqual(
            workload["pause_policy"],
            "zero_attributed_or_unexplained_itl_outliers",
        )

    def test_semantic_hash_excludes_dynamic_envelope_but_binds_output(self) -> None:
        first = stream_result("request", "same")
        second = stream_result("request", "same")
        second.semantic_deltas[0]["id"] = "different-id"
        second.semantic_deltas[0]["created"] = 999
        third = stream_result("request", "different")
        self.assertEqual(
            resilience.canonical_semantic_hash(first),
            resilience.canonical_semantic_hash(second),
        )
        self.assertNotEqual(
            resilience.canonical_semantic_hash(first),
            resilience.canonical_semantic_hash(third),
        )

    def test_budget_event_contract_counts_every_closed_path(self) -> None:
        value = graph(
            budget_evictions=1,
            byte_budget_rejections=2,
            pre_capture_byte_budget_skips=3,
            fallback_graph_cache_byte_budget=4,
        )
        self.assertEqual(resilience.graph_budget_events(value), 10)

    def test_metrics_compare_exact_outputs_and_publish_every_wave(self) -> None:
        results = tuple(
            stream_result(f"resilience-c{level}-r00")
            for level in resilience.CONCURRENCY_LEVELS
        )
        outputs = {
            result.name: resilience.canonical_semantic_hash(result) for result in results
        }
        headroom = resilience.ArmRun(
            "headroom",
            1 << 30,
            results,
            outputs,
            graph(),
            graph(),
            100,
            0,
            0,
        )
        tight = resilience.ArmRun(
            "tight",
            64 << 20,
            results,
            outputs,
            graph(),
            graph(pre_capture_byte_budget_skips=1),
            90,
            0,
            0,
        )
        metrics, details = resilience.metrics_from_arms(
            {"headroom": headroom, "tight": tight}
        )
        by_name = {metric["name"]: metric["value"] for metric in metrics}
        self.assertIsNone(details)
        self.assertEqual(by_name["output_mismatch_count"], 0)
        self.assertEqual(by_name["graph_budget_event_count"], 1)
        for level in resilience.CONCURRENCY_LEVELS:
            self.assertEqual(by_name[f"concurrency_{level}_request_count"], 1)

        drifted = dict(outputs)
        drifted[next(iter(drifted))] = "sha256:different"
        mismatched = dataclasses_replace(tight, outputs=drifted)
        metrics, details = resilience.metrics_from_arms(
            {"headroom": headroom, "tight": mismatched}
        )
        by_name = {metric["name"]: metric["value"] for metric in metrics}
        self.assertEqual(by_name["output_mismatch_count"], 1)
        self.assertIsNotNone(details)

def dataclasses_replace(value: resilience.ArmRun, **changes: object) -> resilience.ArmRun:
    fields = {
        field.name: getattr(value, field.name)
        for field in resilience.dataclasses.fields(value)
    }
    fields.update(changes)
    return resilience.ArmRun(**fields)


if __name__ == "__main__":
    unittest.main()
