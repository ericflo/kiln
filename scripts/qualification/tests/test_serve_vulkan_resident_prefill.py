from __future__ import annotations

import importlib.util
import json
import math
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock


QUALIFICATION_DIR = Path(__file__).resolve().parents[1]
ROOT = QUALIFICATION_DIR.parents[1]
sys.path.insert(0, str(QUALIFICATION_DIR))
SPEC = importlib.util.spec_from_file_location(
    "qualification_serve_vulkan_resident_prefill",
    QUALIFICATION_DIR / "serve_vulkan_resident_prefill.py",
)
assert SPEC is not None and SPEC.loader is not None
oracle = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = oracle
SPEC.loader.exec_module(oracle)


def stream_result(
    *,
    name: str = "resident-c0-r0",
    completion_tokens: int = 8,
    content: str = "000000 ",
    resident_prefill_used: bool = True,
) -> oracle.mixed.StreamResult:
    return oracle.mixed.StreamResult(
        name=name,
        marker="marker",
        started=1.0,
        finished=2.0,
        semantic_times=[1.25],
        token_ready_times=[1.25] * completion_tokens,
        token_queue_delays_ms=[0.0] * completion_tokens,
        prompt_tokens=32,
        completion_tokens=completion_tokens,
        usage_records=1,
        finish_reason="length",
        done=True,
        cancelled=False,
        error=None,
        semantic_deltas=[
            {"choices": [{"index": 0, "delta": {"content": content}}]}
        ],
        resident_prefill_used=resident_prefill_used,
        token_ids=list(range(completion_tokens)),
    )


class ServeVulkanResidentPrefillTests(unittest.TestCase):
    def test_checked_in_workload_exactly_matches_driver_contract(self) -> None:
        manifest = json.loads(
            (
                ROOT
                / "qualification/workloads/serving-vulkan-resident-prefill-v1.json"
            ).read_text()
        )
        self.assertEqual(
            manifest["workload_id"], "serving-vulkan-resident-prefill-v1"
        )
        self.assertEqual(len(manifest["variants"]), 1)
        variant = manifest["variants"][0]
        self.assertEqual(variant["id"], oracle.VARIANT_ID)
        self.assertEqual(variant["backend"], "vulkan")
        self.assertEqual(variant["device_requirement"], "required")
        self.assertEqual(variant["skip_policy"], "fail")
        self.assertEqual(variant["effective_config"], oracle.EFFECTIVE_CONFIG)
        case = variant["cases"][0]
        self.assertEqual(case["id"], oracle.CASE_ID)
        self.assertEqual(
            case["result_protocol"]["declared_metrics"],
            sorted(oracle.METRIC_DEFINITIONS),
        )
        self.assertEqual(
            case["command"],
            [
                "python3",
                "scripts/qualification/serve_vulkan_resident_prefill.py",
                "--model-path",
                "${model_path}",
                "--seed",
                "${seed}",
            ],
        )

    def test_oracle_uses_repeated_equal_prompt_changing_cohorts(self) -> None:
        self.assertEqual(oracle.COHORT_MAX_TOKENS, ((8, 12, 16, 20), (20, 16, 12, 8)))
        self.assertEqual(oracle.EXPECTED_REQUESTS, 8)
        self.assertEqual(oracle.EXPECTED_COMPLETION_TOKENS, 112)
        workload = oracle.EFFECTIVE_CONFIG["workload"]
        self.assertEqual(workload["cohort_prompt_words"], 16)
        self.assertEqual(workload["same_process_repetitions"], 2)
        self.assertTrue(workload["varying_completion_lengths"])
        self.assertEqual(
            workload["simultaneous_dispatch"], "per_cohort_thread_barrier"
        )
        self.assertEqual(
            oracle.EFFECTIVE_CONFIG["runtime"]["serving_profile"], "experimental"
        )

    def test_source_build_and_host_memory_bounds_are_closed(self) -> None:
        build = oracle.EFFECTIVE_CONFIG["build"]
        self.assertEqual(build["features"], "vulkan")
        self.assertEqual(build["cargo_wrapper"], "scripts/cargo-bounded.sh")
        self.assertEqual(build["cargo_min_available_gib"], 1)
        self.assertEqual(build["timeout_seconds"], 900)
        workload = oracle.EFFECTIVE_CONFIG["workload"]
        self.assertEqual(workload["host_mem_available_floor_bytes"], 8 << 30)
        self.assertEqual(workload["host_swap_growth_limit_bytes"], 512 << 20)

    def test_cohort_runner_delivers_slot_specific_completion_limits(self) -> None:
        limits = (8, 12, 16, 20)

        def fake_stream(_port: int, **kwargs: object) -> oracle.mixed.StreamResult:
            return stream_result(
                name=str(kwargs["name"]),
                completion_tokens=int(kwargs["max_tokens"]),
            )

        with mock.patch.object(oracle.mixed, "run_stream", side_effect=fake_stream) as run:
            cohort = oracle.run_cohort(
                8420,
                cohort_index=0,
                max_tokens=limits,
                seed=7,
                absolute_deadline=oracle.time.monotonic() + 10,
            )
        self.assertEqual(
            tuple(result.completion_tokens for result in cohort.results), limits
        )
        self.assertEqual(
            sorted(call.kwargs["max_tokens"] for call in run.call_args_list),
            list(limits),
        )

    def test_result_contract_rejects_corruption_and_wrong_length(self) -> None:
        valid = [
            stream_result(name=f"resident-c0-r{index}", completion_tokens=limit)
            for index, limit in enumerate(oracle.COHORT_MAX_TOKENS[0])
        ]
        valid.extend(
            stream_result(name=f"resident-c1-r{index}", completion_tokens=limit)
            for index, limit in enumerate(oracle.COHORT_MAX_TOKENS[1])
        )
        cohorts = [
            oracle.CohortRun(0, tuple(valid[:4]), 1.0, 2.0),
            oracle.CohortRun(1, tuple(valid[4:]), 2.0, 3.0),
        ]
        self.assertEqual(oracle.result_failures(cohorts), [])

        cohorts[1].results[0].semantic_deltas[0]["choices"][0]["delta"][
            "content"
        ] = "corrupt"
        failures = oracle.result_failures(cohorts)
        self.assertTrue(any("response oracle failed" in failure for failure in failures))

    def test_metric_records_are_closed_sorted_and_finite(self) -> None:
        values = {name: index for index, name in enumerate(oracle.METRIC_DEFINITIONS)}
        records = oracle.metric_records(values)
        self.assertEqual(
            [record["name"] for record in records],
            sorted(oracle.METRIC_DEFINITIONS),
        )
        values["resident_prefill_max_batch_size"] = math.nan
        with self.assertRaisesRegex(
            oracle.ResidentPrefillOracleError, "not finite"
        ):
            oracle.metric_records(values)

    def test_execution_identity_binds_vulkan_binary(self) -> None:
        digest = "sha256:" + "a" * 64
        health = {
            "execution_identity": {
                "backend": "vulkan",
                "device": "vulkan:0",
                "executable_sha256": digest,
            }
        }
        self.assertEqual(oracle.execution_identity_failures(health, digest), [])
        health["execution_identity"]["executable_sha256"] = "sha256:" + "b" * 64
        self.assertTrue(oracle.execution_identity_failures(health, digest))

    def test_main_publishes_structured_result(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            model = root / "model"
            result_path = root / "result.json"
            model.mkdir()
            with (
                mock.patch.dict(
                    os.environ,
                    {
                        oracle.VARIANT_ENV: oracle.VARIANT_ID,
                        oracle.RESULT_ENV: str(result_path),
                    },
                ),
                mock.patch.object(oracle, "execute", return_value=None),
            ):
                exit_code = oracle.main(
                    ["--model-path", str(model), "--seed", "7"]
                )
            self.assertEqual(exit_code, 0)
            result = json.loads(result_path.read_text())
            self.assertEqual(result["case_id"], oracle.CASE_ID)
            self.assertEqual(result["status"], "passed")
            self.assertEqual(result["effective_config"], oracle.EFFECTIVE_CONFIG)
            self.assertEqual(
                [metric["name"] for metric in result["metrics"]],
                sorted(oracle.METRIC_DEFINITIONS),
            )


if __name__ == "__main__":
    unittest.main()
