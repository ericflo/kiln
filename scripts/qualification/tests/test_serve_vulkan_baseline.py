from __future__ import annotations

import importlib.util
import json
import math
import os
import sys
import tempfile
import threading
import unittest
from pathlib import Path
from unittest import mock


QUALIFICATION_DIR = Path(__file__).resolve().parents[1]
ROOT = QUALIFICATION_DIR.parents[1]
sys.path.insert(0, str(QUALIFICATION_DIR))
SPEC = importlib.util.spec_from_file_location(
    "qualification_serve_vulkan_baseline",
    QUALIFICATION_DIR / "serve_vulkan_baseline.py",
)
assert SPEC is not None and SPEC.loader is not None
baseline = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = baseline
SPEC.loader.exec_module(baseline)


def stream_result(content: str) -> baseline.mixed.StreamResult:
    return baseline.mixed.StreamResult(
        name="single-00",
        marker="marker",
        started=1.0,
        finished=2.0,
        semantic_times=[1.25],
        token_ready_times=[1.25, 1.5],
        token_queue_delays_ms=[0.0, 0.0],
        prompt_tokens=4,
        completion_tokens=2,
        usage_records=1,
        finish_reason="length",
        done=True,
        cancelled=False,
        error=None,
        semantic_deltas=[
            {
                "id": "dynamic",
                "created": 1,
                "choices": [{"index": 0, "delta": {"content": content}}],
            }
        ],
    )


def identity_fixture() -> tuple[dict, dict, str]:
    digest = "sha256:" + "a" * 64
    provenance = {
        "provenance_type": "kiln.execution-provenance.v1",
        "provenance_sha256": digest,
        "backend": {
            "name": "vulkan",
            "device": "vulkan:0",
            "numerical_runtime_sha256": digest,
        },
        "build": {"executable_sha256": digest, "source_dirty": False},
        "kernels": {
            "contract_sha256": digest,
            "compiled_features": ["vulkan"],
        },
        "precision": {"inference_dtype": "bf16", "training_policy": "f32"},
        "configuration": {
            "effective_server_config_sha256": digest,
            "effective_environment_sha256": digest,
        },
    }
    identity = {
        "provenance_type": provenance["provenance_type"],
        "provenance_sha256": digest,
        "backend": "vulkan",
        "device": "vulkan:0",
        "executable_sha256": digest,
        "numerical_runtime_sha256": digest,
        "kernel_contract_sha256": digest,
        "inference_dtype": "bf16",
        "training_policy": "f32",
        "effective_server_config_sha256": digest,
        "effective_environment_sha256": digest,
    }
    return {"execution_identity": identity}, {"model": {"execution_provenance": provenance}}, digest


class ServeVulkanBaselineTests(unittest.TestCase):
    def test_checked_in_workload_exactly_matches_driver_contract(self) -> None:
        manifest = json.loads(
            (
                ROOT / "qualification/workloads/serving-vulkan-baseline-v1.json"
            ).read_text()
        )
        self.assertEqual(manifest["workload_id"], "serving-vulkan-baseline-v1")
        self.assertEqual(len(manifest["variants"]), 1)
        variant = manifest["variants"][0]
        self.assertEqual(variant["id"], baseline.VARIANT_ID)
        self.assertEqual(variant["backend"], "vulkan")
        self.assertEqual(variant["device_requirement"], "required")
        self.assertEqual(variant["skip_policy"], "fail")
        self.assertEqual(variant["effective_config"], baseline.EFFECTIVE_CONFIG)
        case = variant["cases"][0]
        self.assertEqual(case["id"], baseline.CASE_ID)
        self.assertEqual(
            case["result_protocol"]["declared_metrics"],
            sorted(baseline.METRIC_DEFINITIONS),
        )
        self.assertEqual(
            case["command"],
            [
                "python3",
                "scripts/qualification/serve_vulkan_baseline.py",
                "--model-path",
                "${model_path}",
                "--seed",
                "${seed}",
            ],
        )
        self.assertNotIn("ROCM_PATH", case["environment"])

    def test_concurrency_sweep_closes_single_through_saturation(self) -> None:
        self.assertEqual([len(words) for _, words in baseline.WAVES], [1, 4, 8, 12])
        self.assertEqual(baseline.EXPECTED_REQUESTS, 25)
        self.assertEqual(baseline.EXPECTED_COMPLETION_TOKENS, 800)
        workload = baseline.EFFECTIVE_CONFIG["workload"]
        self.assertEqual(workload["simultaneous_dispatch"], "per_wave_thread_barrier")
        self.assertEqual(workload["pause_gate_ms"], 2_000)

    def test_source_build_is_vulkan_only_bounded_and_timeout_separated(self) -> None:
        build = baseline.EFFECTIVE_CONFIG["build"]
        self.assertEqual(build["features"], "vulkan")
        self.assertEqual(build["cargo_wrapper"], "scripts/cargo-bounded.sh")
        self.assertEqual(build["cargo_min_available_gib"], 15)
        self.assertEqual(build["timeout_seconds"], 900)
        self.assertNotIn("rocm_path", build)
        self.assertNotIn("rocm_archs", build)
        with mock.patch.object(
            baseline.mixed,
            "build_binary",
            side_effect=baseline.VulkanBaselineError("build boundary reached"),
        ) as build_binary:
            evidence = baseline.RunEvidence()
            with self.assertRaisesRegex(
                baseline.VulkanBaselineError, "build boundary reached"
            ):
                baseline.execute(Path("/model"), 7, evidence)
        self.assertIs(build_binary.call_args.args[1], baseline.mixed.VULKAN_BUILD_SPEC)
        self.assertEqual(
            build_binary.call_args.kwargs["build_timeout_seconds"],
            baseline.BUILD_TIMEOUT_SECONDS,
        )

    def test_semantic_hash_excludes_dynamic_envelope_and_binds_content(self) -> None:
        first = stream_result("same")
        second = stream_result("same")
        second.semantic_deltas[0]["id"] = "other"
        second.semantic_deltas[0]["created"] = 999
        third = stream_result("different")
        first_wave = (baseline.WaveRun("single", (first,), 1.0, 2.0),)
        second_wave = (baseline.WaveRun("single", (second,), 1.0, 2.0),)
        third_wave = (baseline.WaveRun("single", (third,), 1.0, 2.0),)
        self.assertEqual(
            baseline.canonical_semantic_sha256(first_wave),
            baseline.canonical_semantic_sha256(second_wave),
        )
        self.assertNotEqual(
            baseline.canonical_semantic_sha256(first_wave),
            baseline.canonical_semantic_sha256(third_wave),
        )

    def test_execution_identity_is_exact_and_source_bound(self) -> None:
        health, debug, binary_sha256 = identity_fixture()
        self.assertEqual(
            baseline.execution_identity_failures(health, debug, binary_sha256), []
        )
        health["execution_identity"]["device"] = "cpu"
        failures = baseline.execution_identity_failures(
            health, debug, binary_sha256
        )
        self.assertTrue(any("disagrees" in failure for failure in failures))
        self.assertTrue(any("vulkan:0" in failure for failure in failures))

    def test_execution_identity_rejects_rocm_feature_and_dirty_source(self) -> None:
        health, debug, binary_sha256 = identity_fixture()
        provenance = debug["model"]["execution_provenance"]
        provenance["kernels"]["compiled_features"].append("rocm")
        provenance["build"]["source_dirty"] = True
        failures = baseline.execution_identity_failures(
            health, debug, binary_sha256
        )
        self.assertTrue(any("includes ROCm" in failure for failure in failures))
        self.assertTrue(any("source_dirty" in failure for failure in failures))

    def test_metric_records_are_closed_sorted_and_finite(self) -> None:
        values = {name: index for index, name in enumerate(baseline.METRIC_DEFINITIONS)}
        records = baseline.metric_records(values)
        self.assertEqual(
            [record["name"] for record in records],
            sorted(baseline.METRIC_DEFINITIONS),
        )
        values["itl_pause_count"] = math.nan
        with self.assertRaisesRegex(baseline.VulkanBaselineError, "not finite"):
            baseline.metric_records(values)

    def test_wave_runner_releases_every_request_through_one_dispatch_gate(self) -> None:
        prompt_words = (16, 32, 64)
        gate = threading.Barrier(len(prompt_words) + 1)
        with (
            mock.patch.object(baseline.threading, "Barrier", return_value=gate) as barrier,
            mock.patch.object(
                baseline.mixed,
                "run_stream",
                side_effect=lambda _port, **kwargs: stream_result(kwargs["name"]),
            ) as run_stream,
        ):
            wave = baseline.run_wave(
                8420,
                2,
                "batch-3",
                prompt_words,
                11,
                baseline.time.monotonic() + 10,
            )
        barrier.assert_called_once_with(4)
        self.assertEqual(len(wave.results), 3)
        self.assertEqual(run_stream.call_count, 3)

    def test_run_evidence_retains_each_completed_wave(self) -> None:
        evidence = baseline.RunEvidence()
        result = stream_result("partial")
        evidence.record_wave(
            baseline.WaveRun("single", (result,), result.started, result.finished)
        )
        self.assertEqual(evidence.values["single_request_count"], 1)
        self.assertEqual(evidence.values["single_completion_token_count"], 2)
        self.assertEqual(evidence.values["request_count"], 1)
        self.assertEqual(evidence.values["request_failure_count"], 0)
        self.assertEqual(evidence.values["semantic_output_record_count"], 1)
        self.assertEqual(
            evidence.details["semantic_output_sha256"],
            baseline.canonical_semantic_sha256(tuple(evidence.waves)),
        )
        self.assertEqual(evidence.milestones, ["wave:single"])

    def test_startup_identity_is_retained_before_measurement(self) -> None:
        evidence = baseline.RunEvidence()
        health, _, _ = identity_fixture()
        baseline.record_execution_identity(evidence, health)
        self.assertEqual(
            evidence.details,
            {
                "effective_environment_sha256": "sha256:" + "a" * 64,
                "effective_server_config_sha256": "sha256:" + "a" * 64,
                "execution_provenance_sha256": "sha256:" + "a" * 64,
                "kernel_contract_sha256": "sha256:" + "a" * 64,
            },
        )

    def test_runtime_observations_retain_startup_device_faults(self) -> None:
        evidence = baseline.RunEvidence()
        evidence.record_runtime_observations(
            [
                baseline.mixed.ObservedEvent(
                    1.0, "device_fault", "device lost during prewarm"
                )
            ],
            [],
            [],
            [],
        )
        self.assertEqual(evidence.values["device_fault_count"], 1)
        self.assertEqual(evidence.values["memory_sample_count"], 0)

    def test_main_publishes_a_structured_result_through_the_shared_writer(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            model = root / "model"
            result_path = root / "result.json"
            model.mkdir()
            with (
                mock.patch.dict(
                    os.environ,
                    {
                        baseline.VARIANT_ENV: baseline.VARIANT_ID,
                        baseline.RESULT_ENV: str(result_path),
                    },
                ),
                mock.patch.object(
                    baseline,
                    "execute",
                    return_value=None,
                ),
            ):
                exit_code = baseline.main(
                    ["--model-path", str(model), "--seed", "7"]
                )
            result = json.loads(result_path.read_text())
        self.assertEqual(exit_code, 0)
        self.assertEqual(result["status"], "passed")
        self.assertEqual(result["case_id"], baseline.CASE_ID)
        self.assertEqual(result["effective_config"], baseline.EFFECTIVE_CONFIG)

    def test_main_serializes_partial_failure_evidence(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            model = root / "model"
            result_path = root / "result.json"
            model.mkdir()

            def fail(
                _model: Path,
                _seed: int,
                evidence: baseline.RunEvidence,
            ) -> None:
                evidence.values["binary_build_count"] = 1
                evidence.details["kiln_binary_sha256"] = "sha256:" + "a" * 64
                evidence.milestones.append("build")
                raise baseline.VulkanBaselineError("synthetic startup failure")

            with (
                mock.patch.dict(
                    os.environ,
                    {
                        baseline.VARIANT_ENV: baseline.VARIANT_ID,
                        baseline.RESULT_ENV: str(result_path),
                    },
                ),
                mock.patch.object(baseline, "execute", side_effect=fail),
            ):
                exit_code = baseline.main(
                    ["--model-path", str(model), "--seed", "7"]
                )

            result = json.loads(result_path.read_text())
        self.assertEqual(exit_code, 1)
        by_name = {
            record["name"]: record["value"] for record in result["metrics"]
        }
        self.assertEqual(by_name["binary_build_count"], 1)
        self.assertEqual(by_name["request_count"], 0)
        self.assertEqual(by_name["request_failure_count"], 0)
        details = json.loads(result["details"])
        self.assertEqual(details["milestones"], ["build"])
        self.assertIn("synthetic startup failure", details["error"])


if __name__ == "__main__":
    unittest.main()
