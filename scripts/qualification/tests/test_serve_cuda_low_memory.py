from __future__ import annotations

import importlib.util
import json
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
    "qualification_serve_cuda_low_memory",
    QUALIFICATION_DIR / "serve_cuda_low_memory.py",
)
assert SPEC is not None and SPEC.loader is not None
low_memory = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = low_memory
SPEC.loader.exec_module(low_memory)


def ready_payload() -> dict:
    return {
        "schema_version": 1,
        "pid": 123,
        "device_ordinal": 0,
        "allocated_bytes": 1024 * low_memory.MIB,
        "allocation_count": 4,
        "target_free_bytes": low_memory.TARGET_FREE_MIB * low_memory.MIB,
        "minimum_free_bytes": low_memory.MINIMUM_FREE_MIB * low_memory.MIB,
        "baseline": {
            "total_bytes": 16 * low_memory.GIB,
            "free_bytes": 3 * low_memory.GIB,
        },
        "ready": {
            "total_bytes": 16 * low_memory.GIB,
            "free_bytes": 2 * low_memory.GIB,
        },
    }


def release_payload() -> dict:
    ready = ready_payload()
    return {
        "schema_version": 1,
        "pid": 123,
        "device_ordinal": 0,
        "ready_written": True,
        "completed": True,
        "allocated_bytes": ready["allocated_bytes"],
        "allocation_count": ready["allocation_count"],
        "minimum_observed_free_bytes": low_memory.MINIMUM_FREE_MIB
        * low_memory.MIB,
        "sample_count": 10,
        "elapsed_seconds": 1.0,
        "release_failures": [],
        "final": {
            "total_bytes": 16 * low_memory.GIB,
            "free_bytes": 3 * low_memory.GIB,
        },
    }


class ServeCudaLowMemoryTests(unittest.TestCase):
    def test_checked_in_workload_exactly_matches_driver_contract(self) -> None:
        path = (
            ROOT
            / "qualification/workloads/serving-cuda-low-memory-v1.json"
        )
        workload = json.loads(path.read_text())
        self.assertEqual(workload["determinism"]["seed_delivery"], "argv")
        self.assertEqual(
            workload["comparison_policy"]["mode"],
            "self_contained_correctness",
        )
        self.assertEqual(len(workload["variants"]), 1)
        variant = workload["variants"][0]
        self.assertEqual(variant["id"], low_memory.VARIANT_ID)
        self.assertEqual(variant["effective_config"], low_memory.EFFECTIVE_CONFIG)
        self.assertEqual(len(variant["cases"]), 1)
        case = variant["cases"][0]
        self.assertEqual(case["id"], low_memory.CASE_ID)
        self.assertEqual(
            case["result_protocol"]["declared_metrics"],
            sorted(low_memory.METRIC_DEFINITIONS),
        )
        self.assertEqual(
            case["command"],
            [
                "python3",
                "scripts/qualification/serve_cuda_low_memory.py",
                "--model-path",
                "${model_path}",
                "--seed",
                "${seed}",
            ],
        )

    def test_closed_toml_parser_accepts_config_and_rejects_composites(self) -> None:
        parsed = low_memory.parse_closed_toml(
            low_memory.SERVER_CONFIG.read_text()
        )
        self.assertEqual(parsed["server"]["port"], 8420)
        self.assertEqual(parsed["memory"]["floor_gb"], 1.5)
        with self.assertRaisesRegex(
            low_memory.mixed.QualificationError, "non-scalar"
        ):
            low_memory.parse_closed_toml("[server]\nports = [8420]\n")
        with self.assertRaisesRegex(
            low_memory.mixed.QualificationError, "duplicate"
        ):
            low_memory.parse_closed_toml("[server]\nport = 1\nport = 2\n")

    def test_prerequisite_receipt_is_exact_and_passed(self) -> None:
        low_memory.validate_admission_prerequisite()
        with mock.patch.object(
            low_memory,
            "ADMISSION_RECEIPT_SHA256",
            "sha256:" + "0" * 64,
        ):
            with self.assertRaisesRegex(
                low_memory.mixed.QualificationError, "hash drifted"
            ):
                low_memory.validate_admission_prerequisite()

    def test_peer_ready_rejects_allocation_above_declared_cap(self) -> None:
        payload = ready_payload()
        payload["allocated_bytes"] = (
            low_memory.PEER_MAX_ALLOCATION_MIB + 1
        ) * low_memory.MIB
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "ready.json"
            path.write_text(json.dumps(payload))
            with self.assertRaisesRegex(
                low_memory.mixed.QualificationError, "allocation cap"
            ):
                low_memory.load_peer_ready(path, 123)

    def test_runner_environment_is_closed_and_child_strips_controls(self) -> None:
        source = {
            "PATH": os.environ["PATH"],
            low_memory.RESULT_ENV: "/tmp/result.json",
            low_memory.VARIANT_ENV: low_memory.VARIANT_ID,
            low_memory.NETWORK_ENV: "util-linux-unshare-user-net-pid-landlock-v1",
            low_memory.THERMAL_POLICY_ENV: "sha256:" + "a" * 64,
            low_memory.SCOPE_BOUNDARY_ENV: "systemd-user-scope-feedback-v1",
            low_memory.SCOPE_MEMORY_MAX_ENV: str(10 * low_memory.GIB),
            low_memory.SCOPE_PIDS_MAX_ENV: "512",
            low_memory.SCOPE_CPU_QUOTA_ENV: "50",
            low_memory.SCOPE_UNIT_ENV: "kiln-wsl-scope-" + "a" * 32,
            low_memory.SCOPE_HOST_UID_ENV: "1000",
        }
        child = low_memory.child_environment(source)
        self.assertEqual(child, {"PATH": os.environ["PATH"]})
        source["KILN_MEMORY_FLOOR_GB"] = "0"
        with self.assertRaisesRegex(
            low_memory.mixed.QualificationError, "ambient Kiln controls"
        ):
            low_memory.child_environment(source)

    def test_ready_payload_is_closed_and_enforces_floor(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "ready.json"
            path.write_text(json.dumps(ready_payload()))
            loaded = low_memory.load_peer_ready(path, 123)
            self.assertEqual(
                loaded["ready"]["free_bytes"], 2 * low_memory.GIB
            )

            malformed = ready_payload()
            malformed["ready"]["free_bytes"] = (
                low_memory.MINIMUM_FREE_MIB * low_memory.MIB - 1
            )
            path.write_text(json.dumps(malformed))
            with self.assertRaisesRegex(
                low_memory.mixed.QualificationError, "crossed"
            ):
                low_memory.load_peer_ready(path, 123)

    def test_release_payload_requires_recovery_and_complete_cleanup(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "release.json"
            path.write_text(json.dumps(release_payload()))
            loaded = low_memory.load_peer_release(
                path, 123, ready_payload()
            )
            self.assertEqual(loaded["sample_count"], 10)

            malformed = release_payload()
            malformed["completed"] = False
            path.write_text(json.dumps(malformed))
            with self.assertRaisesRegex(
                low_memory.mixed.QualificationError, "completed"
            ):
                low_memory.load_peer_release(path, 123, ready_payload())

            malformed = release_payload()
            malformed["final"]["free_bytes"] = 2 * low_memory.GIB
            path.write_text(json.dumps(malformed))
            with self.assertRaisesRegex(
                low_memory.mixed.QualificationError, "did not recover"
            ):
                low_memory.load_peer_release(path, 123, ready_payload())

    def test_live_memory_requires_healthy_nvidia_sampler(self) -> None:
        live = {
            "probe_failed": False,
            "sample_stale": False,
            "sampler_required": True,
            "sampler_running": True,
            "sampler_healthy": True,
            "source": "nvidia-smi",
            "unified": False,
            "free_gb": 2.0,
        }
        health = {"gpu_memory": {"live": live}}
        self.assertEqual(
            low_memory.live_free_bytes(health, "fixture"),
            2_000_000_000,
        )
        live["sample_stale"] = True
        with self.assertRaisesRegex(
            low_memory.mixed.QualificationError, "sample_stale"
        ):
            low_memory.live_free_bytes(health, "fixture")

    def test_model_attestation_binds_residency_and_execution(self) -> None:
        binary_hash = "sha256:" + "b" * 64
        health = {
            "status": "ok",
            "backend": "model",
            "backend_runtime": {
                "healthy": True,
                "quarantined": False,
                "reason": None,
                "restart_required": False,
            },
            "base_weight_identity": {
                "manifest_type": "kiln.base-weight-shards.v1",
                "aggregate_algorithm": "kiln.base-model-content.v1",
                "aggregate_sha256": "sha256:" + "a" * 64,
                "shard_count": 2,
                "total_size_bytes": 9 * low_memory.GIB,
            },
            "gpu_memory": {
                "post_load_used_bytes": 8 * low_memory.GIB,
                "live": {
                    "probe_failed": False,
                    "sample_stale": False,
                    "sampler_required": True,
                    "sampler_running": True,
                    "sampler_healthy": True,
                    "source": "nvidia-smi",
                    "unified": False,
                    "free_gb": 3.0,
                },
            },
        }
        models = {"data": [{"id": low_memory.MODEL_ID}]}
        debug = {
            "model": {
                "execution_provenance": {
                    "provenance_type": "kiln.execution-provenance.v1",
                    "backend": {"name": "cuda", "device": "cuda:0"},
                    "build": {
                        "executable_sha256": binary_hash,
                        "source_dirty": False,
                    },
                }
            }
        }
        self.assertEqual(
            low_memory.attest_model(health, models, debug, binary_hash),
            8 * low_memory.GIB,
        )
        health["base_weight_identity"]["aggregate_sha256"] = "a" * 64
        with self.assertRaisesRegex(
            low_memory.mixed.QualificationError, "base-weight identity"
        ):
            low_memory.attest_model(health, models, debug, binary_hash)

    def test_device_identity_is_exact(self) -> None:
        completed = mock.Mock(
            returncode=0,
            stdout="NVIDIA GeForce RTX 4090 Laptop GPU, 16376\n",
            stderr="",
        )
        with mock.patch.object(
            low_memory.subprocess, "run", return_value=completed
        ) as run:
            low_memory.validate_device_identity()
        self.assertEqual(run.call_args.args[0][0], "/usr/lib/wsl/lib/nvidia-smi")

        completed.stdout = "NVIDIA GeForce RTX 4090, 24564\n"
        with mock.patch.object(
            low_memory.subprocess, "run", return_value=completed
        ):
            with self.assertRaisesRegex(
                low_memory.mixed.QualificationError, "identity drifted"
            ):
                low_memory.validate_device_identity()

    def test_metric_contract_is_closed_sorted_and_finite(self) -> None:
        metrics = low_memory.zero_metrics()
        self.assertEqual(
            [metric["name"] for metric in metrics],
            sorted(low_memory.METRIC_DEFINITIONS),
        )
        values = {name: 0 for name in low_memory.METRIC_DEFINITIONS}
        del values["request_failure_count"]
        with self.assertRaisesRegex(
            low_memory.mixed.QualificationError, "mismatch"
        ):
            low_memory.metrics_from_values(values)


if __name__ == "__main__":
    unittest.main()
