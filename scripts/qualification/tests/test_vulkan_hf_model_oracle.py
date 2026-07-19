from __future__ import annotations

import importlib.util
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path


QUALIFICATION_DIR = Path(__file__).resolve().parents[1]
ROOT = Path(__file__).resolve().parents[3]
SPEC = importlib.util.spec_from_file_location(
    "vulkan_hf_model_oracle",
    QUALIFICATION_DIR / "vulkan_hf_model_oracle.py",
)
assert SPEC is not None and SPEC.loader is not None
oracle = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = oracle
SPEC.loader.exec_module(oracle)


def hf_fixture(**updates):
    value = {
        "memory_limit_gib": 16,
        "logits_sha256": "sha256:" + "c" * 64,
        "memory_high_events": 0,
        "memory_max_events": 0,
        "memory_peak_bytes": 10_000_000_000,
        "reference_sha256": "sha256:" + "a" * 64,
        "swap_bytes": 0,
        "thermal": {
            "phase_settlement_timeout_seconds": 300.0,
            "policy": {
                "content_sha256": "sha256:e7347f6c698b33bf3fd6a1a76483118c61d3e5d8b19b64d45609014da40f7620",
                "id": "strix-halo-hf-oracle-v1",
                "limit_millicelsius": 93000,
                "pacing": {
                    "resume_millicelsius": 50000,
                    "resume_stable_samples": 20,
                    "start_millicelsius": 58000,
                },
                "phase_settlement_timeout_seconds": 300.0,
                "poll_interval_ms": 50,
                "safe_handoff": {
                    "stable_samples": 20,
                    "target_millicelsius": 45000,
                    "timeout_seconds": 300.0,
                },
                "schema": "kiln.host-thermal-policy.v1",
                "sensor": {"hwmon_name": "k10temp", "label": "Tctl"},
            },
            "prelaunch_cooldown": {"completed": True},
            "runtime": {
                "host_temperature_peak_millicelsius": 70000,
                "host_thermal_pacing_event_count": 2,
                "host_thermal_pacing_seconds": 3.5,
            },
            "schema": "kiln.hf-thermal-containment.v1",
            "worker_exit_code": 0,
        },
    }
    value.update(updates)
    return value


class VulkanHfModelOracleTests(unittest.TestCase):
    def test_invocation_artifact_path_is_anchored_without_dereferencing(self) -> None:
        relative = Path(".qualification/results/case.json")
        expected = Path(os.getcwd()) / relative
        self.assertEqual(oracle._absolute_invocation_path(relative), expected)
        self.assertTrue(oracle._absolute_invocation_path(relative).is_absolute())

        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / "target"
            target.mkdir()
            link = Path(directory) / "link"
            link.symlink_to(target, target_is_directory=True)
            anchored = oracle._absolute_invocation_path(link / "new.json")
            self.assertEqual(anchored.parent, link)

    def test_child_processes_reject_relative_artifact_paths(self) -> None:
        with self.assertRaisesRegex(oracle.QualificationError, "paths must be absolute"):
            oracle._run_hf_reference(
                python=Path("python"),
                model=Path("model"),
                output=Path("reference.safetensors"),
                workspace=Path("workspace"),
                policy=Path("policy.json"),
            )
        with self.assertRaisesRegex(oracle.QualificationError, "paths must be absolute"):
            oracle._run_vulkan_comparison(
                model=Path("model"), reference=Path("reference.safetensors")
            )

    def test_memory_limit_matches_manifest_and_refuses_reduced_ceiling(self) -> None:
        self.assertEqual(oracle._bounded_memory_limit_gib(24), 16)
        self.assertEqual(oracle._bounded_memory_limit_gib(23), 16)
        with self.assertRaisesRegex(oracle.QualificationError, "require at least 23 GiB"):
            oracle._bounded_memory_limit_gib(22)

    def test_model_input_rejects_a_final_symlink_before_resolution(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            model = root / "model"
            model.mkdir()
            link = root / "model-link"
            link.symlink_to(model, target_is_directory=True)
            with self.assertRaisesRegex(oracle.QualificationError, "non-symlink"):
                oracle._validate_inputs(link, Path(sys.executable), root / "policy.json")

    def test_hf_service_is_private_bounded_zero_swap_and_environment_empty(self) -> None:
        command = oracle._bounded_hf_command(
            unit="kiln-hf-oracle-bounded-test.service",
            python=Path("/venv/bin/python"),
            model=Path("/models/qwen"),
            output=Path("/run/hf.safetensors"),
            policy=Path("/repo/qualification/host-policies/oracle.json"),
            workspace=Path("/run"),
            temporary_directory=Path("/run/tmp"),
            memory_limit_gib=16,
        )
        for expected in (
            "--wait",
            "--collect",
            "--pipe",
            "MemoryMax=16G",
            "MemorySwapMax=0",
            "KillMode=control-group",
            "RuntimeMaxSec=600s",
            "PrivateNetwork=yes",
            "/usr/bin/env",
            "-i",
            "HF_HUB_OFFLINE=1",
            "TRANSFORMERS_OFFLINE=1",
            "/venv/bin/python",
            str(oracle.HF_SUPERVISOR),
            "/repo/qualification/host-policies/oracle.json",
            "/models/qwen",
            "/run/hf.safetensors",
        ):
            self.assertIn(expected, command)
        joined = "\0".join(command)
        self.assertNotIn("GITHUB_TOKEN", joined)
        self.assertNotIn("HF_TOKEN", joined)
        self.assertNotIn("KILN_QUALIFICATION_MODEL_PATH", joined)

    def test_rust_marker_parser_requires_one_finite_complete_record(self) -> None:
        marker = (
            "KILN_VULKAN_HF_FULL_LOGIT_PASS vocab=248320 argmax_equal=1 "
            "hf_argmax=1 kiln_argmax=1 top10_overlap=10 max_abs=1.25e-1 "
            "mean_abs=2.5e-3 cosine=0.999999\n"
        )
        metrics = oracle._parse_rust_metrics(marker)
        self.assertEqual(metrics["vocab"], 248320)
        self.assertEqual(metrics["argmax_equal"], 1)
        self.assertEqual(metrics["top10_overlap"], 10)
        self.assertAlmostEqual(metrics["max_abs"], 0.125)
        with self.assertRaisesRegex(oracle.QualificationError, "found 0"):
            oracle._parse_rust_metrics("no marker")
        with self.assertRaisesRegex(oracle.QualificationError, "found 2"):
            oracle._parse_rust_metrics(marker + marker)
        with self.assertRaisesRegex(oracle.QualificationError, "non-finite"):
            oracle._parse_rust_metrics(marker.replace("max_abs=1.25e-1", "max_abs=nan"))

    def test_hf_marker_parser_is_closed_and_requires_valid_memory(self) -> None:
        evidence = {
            "argmax": 1,
            "device": "AMD Radeon 8060S Graphics",
            "duration_seconds": 5.0,
            "logits_sha256": "sha256:" + "a" * 64,
            "memory_high_events": 0,
            "memory_max_events": 0,
            "memory_oom_events": 0,
            "memory_oom_kill_events": 0,
            "memory_peak_bytes": 14_000_000_000,
            "memory_swap_bytes": 0,
            "output_bytes": 993_896,
            "torch_hip_version": "7.2.53211",
            "torch_version": "2.13.0+rocm7.2",
            "transformers_version": "5.13.1",
            "vocab": 248_320,
        }
        marker = oracle.HF_PASS_PREFIX + json.dumps(evidence)
        self.assertEqual(oracle._parse_hf_evidence(marker), evidence)
        with self.assertRaisesRegex(oracle.QualificationError, "found 2"):
            oracle._parse_hf_evidence(marker + "\n" + marker)
        evidence["unexpected"] = 1
        with self.assertRaisesRegex(oracle.QualificationError, "not closed"):
            oracle._parse_hf_evidence(
                oracle.HF_PASS_PREFIX + json.dumps(evidence)
            )

    def test_case_result_is_closed_sorted_and_records_thresholds(self) -> None:
        result = oracle._result_document(
            duration=12.5,
            hf=hf_fixture(),
            comparison={
                "argmax_equal": 1,
                "cosine": 0.99999,
                "max_abs": 0.25,
                "mean_abs": 0.01,
                "top10_overlap": 10,
                "vocab": 248_320,
            },
        )
        metric_names = [metric["name"] for metric in result["metrics"]]
        self.assertEqual(metric_names, sorted(metric_names))
        self.assertEqual(
            metric_names,
            [
                "argmax_equal",
                "cosine_similarity",
                "hf_host_temperature_peak_millicelsius",
                "hf_peak_memory_bytes",
                "hf_swap_bytes",
                "hf_thermal_pacing_event_count",
                "hf_thermal_pacing_seconds",
                "max_abs_error",
                "mean_abs_error",
                "top10_overlap",
                "vocab_logits",
            ],
        )
        tolerances = {item["metric"]: item for item in result["tolerances"]}
        self.assertEqual(tolerances["max_abs_error"]["absolute_tolerance"], 0.5)
        self.assertEqual(tolerances["mean_abs_error"]["absolute_tolerance"], 0.05)
        self.assertEqual(result["effective_config"]["hf_memory_max_gib"], 16)
        self.assertEqual(result["effective_config"]["kiln_memory_max_gib"], 17)
        self.assertIn("hf_reference_sha256", result["details"])

    def test_declared_workload_config_matches_command_result(self) -> None:
        workload = json.loads(
            (ROOT / "qualification/workloads/vulkan-hf-full-model-oracle-v1.json").read_text()
        )
        variant = workload["variants"][0]
        result = oracle._result_document(
            duration=1.0,
            hf=hf_fixture(
                logits_sha256="sha256:" + "d" * 64,
                memory_peak_bytes=1,
                reference_sha256="sha256:" + "b" * 64,
            ),
            comparison={
                "argmax_equal": 1,
                "cosine": 1.0,
                "max_abs": 0.0,
                "mean_abs": 0.0,
                "top10_overlap": 10,
                "vocab": 248_320,
            },
        )
        self.assertEqual(result["effective_config"], variant["effective_config"])
        case = next(case for case in variant["cases"] if case["id"] == oracle.CASE_ID)
        self.assertEqual(
            [metric["name"] for metric in result["metrics"]],
            case["result_protocol"]["declared_metrics"],
        )


if __name__ == "__main__":
    unittest.main()
