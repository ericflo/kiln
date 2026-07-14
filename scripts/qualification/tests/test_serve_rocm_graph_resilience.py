from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
import time
import unittest
from pathlib import Path
from unittest import mock


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
    token_times = [1.1 + index * 0.1 for index in range(resilience.MAX_TOKENS)]
    return resilience.mixed.StreamResult(
        name=name,
        marker="marker",
        started=1.0,
        finished=2.0,
        semantic_times=[1.25],
        token_ready_times=token_times,
        token_queue_delays_ms=[0.0] * len(token_times),
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
        self.assertEqual(workload["request_timeout_seconds"], 300)
        self.assertEqual(
            resilience.EFFECTIVE_CONFIG["server"]["request_timeout_seconds"],
            360,
        )

    def test_wave_delivers_the_declared_correctness_timeout(self) -> None:
        expected = stream_result("resilience-c1-r00")
        with mock.patch.object(
            resilience.mixed,
            "run_stream",
            return_value=expected,
        ) as run_stream:
            wave = resilience.run_wave(
                12345,
                "headroom",
                1,
                7,
                time.monotonic() + 1.0,
            )
        self.assertEqual(wave, [expected])
        self.assertEqual(
            run_stream.call_args.kwargs["request_timeout_seconds"],
            resilience.REQUEST_TIMEOUT_SECONDS,
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
            self.assertEqual(
                by_name[f"concurrency_{level}_attempted_request_count"], 1
            )
            self.assertEqual(by_name[f"concurrency_{level}_failure_count"], 0)
        self.assertEqual(by_name["headroom_arm_completed"], 1)
        self.assertEqual(by_name["headroom_arm_started"], 1)
        self.assertEqual(by_name["tight_arm_completed"], 1)
        self.assertEqual(by_name["tight_arm_started"], 1)
        self.assertEqual(by_name["tight_attempted_request_count"], 5)
        self.assertEqual(by_name["tight_completed_request_count"], 5)
        self.assertEqual(by_name["case_failure_count"], 0)

        drifted = dict(outputs)
        drifted[next(iter(drifted))] = "sha256:different"
        mismatched = dataclasses_replace(tight, outputs=drifted)
        metrics, details = resilience.metrics_from_arms(
            {"headroom": headroom, "tight": mismatched}
        )
        by_name = {metric["name"]: metric["value"] for metric in metrics}
        self.assertEqual(by_name["output_mismatch_count"], 1)
        self.assertIsNotNone(details)

    def test_partial_failure_metrics_retain_completed_wave_evidence(self) -> None:
        evidence = resilience.RunEvidence()
        evidence.arm_started.add("headroom")
        evidence.record_graph("headroom", graph(capture_successes=9, replay_successes=31))
        evidence.record_peak_memory("headroom", 1234)
        evidence.record_wave("headroom", 1, [stream_result("resilience-c1-r00")])

        failed = resilience.dataclasses.replace(
            stream_result("resilience-c64-r01"),
            semantic_times=[],
            token_ready_times=[],
            token_queue_delays_ms=[],
            prompt_tokens=0,
            completion_tokens=0,
            usage_records=0,
            finish_reason=None,
            done=False,
            error="TimeoutError: timed out",
        )
        evidence.record_wave(
            "headroom",
            64,
            [stream_result("resilience-c64-r00"), failed],
        )
        evidence.case_failure_count = 1

        metrics, details = resilience.metrics_from_evidence(evidence)
        by_name = {metric["name"]: metric["value"] for metric in metrics}
        self.assertIsNone(details)
        self.assertEqual(by_name["case_failure_count"], 1)
        self.assertEqual(by_name["request_failure_count"], 1)
        self.assertEqual(by_name["headroom_arm_completed"], 0)
        self.assertEqual(by_name["headroom_arm_started"], 1)
        self.assertEqual(by_name["tight_arm_completed"], 0)
        self.assertEqual(by_name["tight_arm_started"], 0)
        self.assertEqual(by_name["tight_attempted_request_count"], 0)
        self.assertEqual(by_name["tight_completed_request_count"], 0)
        self.assertEqual(by_name["headroom_attempted_request_count"], 3)
        self.assertEqual(by_name["headroom_completed_request_count"], 2)
        self.assertEqual(by_name["concurrency_1_attempted_request_count"], 1)
        self.assertEqual(by_name["concurrency_1_request_count"], 1)
        self.assertEqual(by_name["concurrency_1_failure_count"], 0)
        self.assertEqual(by_name["concurrency_64_attempted_request_count"], 2)
        self.assertEqual(by_name["concurrency_64_request_count"], 1)
        self.assertEqual(by_name["concurrency_64_failure_count"], 1)
        self.assertEqual(by_name["max_completed_concurrency"], 1)
        self.assertEqual(by_name["headroom_graph_capture_count"], 9)
        self.assertEqual(by_name["headroom_graph_replay_count"], 31)
        self.assertEqual(by_name["headroom_peak_gpu_memory_used_bytes"], 1234)

    def test_main_writes_partial_evidence_when_execution_raises(self) -> None:
        def fail_after_wave(
            _model_path: Path,
            _seed: int,
            evidence: resilience.RunEvidence,
        ) -> None:
            evidence.arm_started.add("headroom")
            evidence.record_graph("headroom", graph(capture_successes=9))
            evidence.record_wave(
                "headroom",
                1,
                [stream_result("resilience-c1-r00")],
            )
            raise resilience.ResilienceError("synthetic failure")

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            model = root / "model"
            model.mkdir()
            result_path = root / "result.json"
            environment = {
                resilience.RESULT_ENV: str(result_path),
                resilience.VARIANT_ENV: resilience.VARIANT_ID,
            }
            with (
                mock.patch.dict(resilience.os.environ, environment, clear=False),
                mock.patch.object(resilience, "execute", side_effect=fail_after_wave),
            ):
                returncode = resilience.main(
                    ["--model-path", str(model), "--seed", "7"]
                )

            self.assertEqual(returncode, 1)
            result = json.loads(result_path.read_text())
            self.assertEqual(result["status"], "failed")
            self.assertIn("synthetic failure", result["details"])
            by_name = {
                metric["name"]: metric["value"] for metric in result["metrics"]
            }
            self.assertEqual(by_name["case_failure_count"], 1)
            self.assertEqual(by_name["headroom_attempted_request_count"], 1)
            self.assertEqual(by_name["headroom_completed_request_count"], 1)
            self.assertEqual(by_name["headroom_graph_capture_count"], 9)

def dataclasses_replace(value: resilience.ArmRun, **changes: object) -> resilience.ArmRun:
    fields = {
        field.name: getattr(value, field.name)
        for field in resilience.dataclasses.fields(value)
    }
    fields.update(changes)
    return resilience.ArmRun(**fields)


if __name__ == "__main__":
    unittest.main()
