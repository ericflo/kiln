from __future__ import annotations

import importlib.util
import json
import math
import sys
import unittest
from pathlib import Path


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
        source = (QUALIFICATION_DIR / "serve_vulkan_baseline.py").read_text()
        self.assertIn("mixed.VULKAN_BUILD_SPEC", source)
        self.assertIn("build_timeout_seconds=BUILD_TIMEOUT_SECONDS", source)
        self.assertNotIn("ROCM_PATH", source)

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

    def test_result_writer_uses_atomic_shared_protocol(self) -> None:
        source = (QUALIFICATION_DIR / "serve_vulkan_baseline.py").read_text()
        self.assertIn("mixed.write_result(Path(result_path_value), result)", source)
        self.assertIn("mixed.terminate_process(process)", source)
        self.assertIn("mixed.snapshot_payload_residue(snapshot_dir)", source)
        self.assertIn("threading.Barrier(len(prompt_words) + 1)", source)


if __name__ == "__main__":
    unittest.main()
