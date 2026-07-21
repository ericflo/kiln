from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock


QUALIFICATION_DIR = Path(__file__).resolve().parents[1]
ROOT = QUALIFICATION_DIR.parents[1]
sys.path.insert(0, str(QUALIFICATION_DIR))
SPEC = importlib.util.spec_from_file_location(
    "qualification_serve_rocm_public_mutation_lifecycle",
    QUALIFICATION_DIR / "serve_rocm_public_mutation_lifecycle.py",
)
assert SPEC is not None and SPEC.loader is not None
lifecycle = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = lifecycle
SPEC.loader.exec_module(lifecycle)


def stream_result(content: str) -> lifecycle.mixed.StreamResult:
    return lifecycle.mixed.StreamResult(
        name="request",
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


class ServeRocmPublicMutationLifecycleTests(unittest.TestCase):
    def test_checked_in_workload_exactly_matches_driver_contract(self) -> None:
        manifest = json.loads(
            (
                ROOT
                / "qualification/workloads/serving-rocm-public-mutation-lifecycle-v1.json"
            ).read_text()
        )
        self.assertEqual(
            manifest["workload_id"],
            "serving-rocm-public-mutation-lifecycle-v1",
        )
        self.assertEqual(len(manifest["variants"]), 1)
        variant = manifest["variants"][0]
        self.assertEqual(variant["id"], lifecycle.VARIANT_ID)
        self.assertEqual(variant["effective_config"], lifecycle.EFFECTIVE_CONFIG)
        self.assertEqual(variant["device_requirement"], "required")
        self.assertEqual(variant["skip_policy"], "fail")
        case = variant["cases"][0]
        self.assertEqual(case["id"], lifecycle.CASE_ID)
        self.assertEqual(
            case["result_protocol"]["declared_metrics"],
            sorted(lifecycle.METRIC_DEFINITIONS),
        )
        self.assertEqual(
            case["command"],
            [
                "python3",
                "scripts/qualification/serve_rocm_public_mutation_lifecycle.py",
                "--model-path",
                "${model_path}",
                "--adapter-path",
                "${adapter_path}",
                "--seed",
                "${seed}",
            ],
        )
        self.assertEqual(manifest["variables"][0]["name"], "adapter_path")
        self.assertTrue(manifest["variables"][0]["required"])

    def test_effective_config_closes_both_public_mutation_arms(self) -> None:
        runtime = lifecycle.EFFECTIVE_CONFIG["runtime"]
        adapter = runtime["adapter_arm"]
        maintenance = runtime["maintenance_resize_arm"]
        self.assertEqual(adapter["serving_profile"], "experimental")
        self.assertTrue(adapter["rocm_graphs_enabled"])
        self.assertFalse(adapter["kv_autoscale_enabled"])
        self.assertEqual(maintenance["serving_profile"], "maintenance")
        self.assertFalse(maintenance["rocm_graphs_enabled"])
        self.assertTrue(maintenance["kv_autoscale_enabled"])
        self.assertEqual(
            maintenance["kv_force_blocks"], lifecycle.FORCED_KV_BLOCKS
        )
        self.assertEqual(
            lifecycle.EFFECTIVE_CONFIG["workload"]["build_reuse"],
            "one_source_bound_binary_for_both_arms",
        )
        self.assertEqual(
            lifecycle.EFFECTIVE_CONFIG["workload"]["adapter_reload"],
            "same_revision_between_requests_barrier",
        )
        self.assertEqual(
            lifecycle.EFFECTIVE_CONFIG["workload"][
                "adapter_overlap_active_max_tokens"
            ],
            lifecycle.OVERLAP_ACTIVE_MAX_TOKENS,
        )

    def test_semantic_hash_ignores_dynamic_envelope_but_binds_content(self) -> None:
        first = stream_result("same")
        second = stream_result("same")
        second.semantic_deltas[0]["id"] = "other"
        second.semantic_deltas[0]["created"] = 999
        third = stream_result("different")
        self.assertEqual(
            lifecycle.canonical_semantic_hash(first),
            lifecycle.canonical_semantic_hash(second),
        )
        self.assertNotEqual(
            lifecycle.canonical_semantic_hash(first),
            lifecycle.canonical_semantic_hash(third),
        )

    def test_copy_adapter_hashes_source_and_private_regular_files(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "source"
            target = root / "target"
            source.mkdir()
            target.mkdir()
            (source / "adapter_config.json").write_text('{"r":8}')
            weights = b"safe tensor fixture"
            (source / "adapter_model.safetensors").write_bytes(weights)
            config_hash, weights_hash, weights_bytes = lifecycle.copy_adapter(
                source, target
            )
            private = target / lifecycle.ADAPTER_NAME
            self.assertEqual(
                config_hash,
                lifecycle.mixed.sha256_file(source / "adapter_config.json"),
            )
            self.assertEqual(
                weights_hash,
                lifecycle.mixed.sha256_file(source / "adapter_model.safetensors"),
            )
            self.assertEqual(weights_bytes, len(weights))
            for name in lifecycle.ADAPTER_FILES:
                self.assertEqual(
                    lifecycle.mixed.sha256_file(source / name),
                    lifecycle.mixed.sha256_file(private / name),
                )

    def test_copy_adapter_rejects_symlinked_inputs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "source"
            target = root / "target"
            source.mkdir()
            target.mkdir()
            backing = root / "config"
            backing.write_text("{}")
            (source / "adapter_config.json").symlink_to(backing)
            (source / "adapter_model.safetensors").write_bytes(b"weights")
            with self.assertRaisesRegex(lifecycle.LifecycleError, "regular file"):
                lifecycle.copy_adapter(source, target)

    def test_adapter_state_requires_revision_agreement_on_every_surface(self) -> None:
        revision = "a" * 64
        health = {
            "loaded_adapter": lifecycle.ADAPTER_NAME,
            "loaded_adapter_revision": revision,
            "loaded_adapter_count": 1,
        }
        debug = {
            "adapters": {
                "loaded_adapter": lifecycle.ADAPTER_NAME,
                "loaded_adapter_revision": revision,
            }
        }
        listed = {
            "loaded_adapter": lifecycle.ADAPTER_NAME,
            "loaded_adapter_identity": {
                "name": lifecycle.ADAPTER_NAME,
                "content_revision": revision,
            },
        }
        with mock.patch.object(
            lifecycle.mixed,
            "json_request",
            side_effect=[health, debug, listed],
        ):
            lifecycle.adapter_state(1234, lifecycle.ADAPTER_NAME, revision)

        listed["loaded_adapter_identity"]["content_revision"] = "b" * 64
        with mock.patch.object(
            lifecycle.mixed,
            "json_request",
            side_effect=[health, debug, listed],
        ):
            with self.assertRaisesRegex(lifecycle.LifecycleError, "identity"):
                lifecycle.adapter_state(1234, lifecycle.ADAPTER_NAME, revision)

    def test_request_phase_preserves_null_and_requires_finite_nonnegative_ms(self) -> None:
        result = stream_result("same")
        result.latency_phases = {"adapter_ms": None}
        self.assertIsNone(lifecycle.request_phase_ms(result, "adapter"))
        result.latency_phases["adapter_ms"] = 12.5
        self.assertEqual(lifecycle.request_phase_ms(result, "adapter"), 12.5)
        for invalid in (-1.0, float("nan"), True, "12"):
            result.latency_phases["adapter_ms"] = invalid
            with self.assertRaisesRegex(lifecycle.LifecycleError, "invalid adapter_ms"):
                lifecycle.request_phase_ms(result, "adapter")

    def test_base_identity_uses_the_shared_normalized_header_contract(self) -> None:
        result = stream_result("same")
        lifecycle.assert_base_response_identity(result, "base")

        result.loaded_adapter = "fixture-adapter"
        result.loaded_adapter_revision = "a" * 64
        with self.assertRaisesRegex(lifecycle.LifecycleError, "named adapter identity"):
            lifecycle.assert_base_response_identity(result, "base")

    def test_wait_actor_adapter_barrier_requires_typed_exclusive_signal(self) -> None:
        health = [
            {
                "sequence": 1,
                "decode_runtime": {
                    "batching_engine": {
                        "actor_barrier_adapter_active": False,
                        "actor_barrier_resize_active": False,
                    }
                },
            },
            {
                "sequence": 2,
                "decode_runtime": {
                    "batching_engine": {
                        "actor_barrier_adapter_active": True,
                        "actor_barrier_resize_active": False,
                    }
                },
            },
        ]
        with (
            mock.patch.object(
                lifecycle.mixed,
                "json_request",
                side_effect=health,
            ),
            mock.patch.object(
                lifecycle.mixed,
                "batching_snapshot",
                return_value={},
            ),
            mock.patch.object(lifecycle.time, "sleep"),
        ):
            observed = lifecycle.wait_actor_adapter_barrier(
                1234,
                lifecycle.time.monotonic() + 1.0,
            )
        self.assertEqual(observed, health[1])

        with (
            mock.patch.object(
                lifecycle.mixed,
                "json_request",
                return_value={
                    "sequence": 3,
                    "decode_runtime": {
                        "batching_engine": {
                            "actor_barrier_adapter_active": True,
                            "actor_barrier_resize_active": True,
                        }
                    },
                },
            ),
            mock.patch.object(
                lifecycle.mixed,
                "batching_snapshot",
                return_value={},
            ),
        ):
            with self.assertRaisesRegex(lifecycle.LifecycleError, "overlapped"):
                lifecycle.wait_actor_adapter_barrier(
                    1234,
                    lifecycle.time.monotonic() + 1.0,
                )

    def test_maintenance_readiness_accepts_only_structured_503_health(self) -> None:
        health = {
            "status": "maintenance",
            "checks": [
                {"name": "inference_admission", "pass": False},
                {"name": "inference_prewarm_complete", "pass": True},
                {"name": "model_loaded", "pass": True},
            ],
        }
        process = mock.Mock()
        process.poll.return_value = None
        server_log = mock.Mock()
        with mock.patch.object(
            lifecycle,
            "json_response",
            return_value=(503, {}, health),
        ):
            observed = lifecycle.wait_maintenance_ready(
                1234,
                process,
                server_log,
                lifecycle.time.monotonic() + 1.0,
            )
        self.assertIs(observed, health)

        with mock.patch.object(
            lifecycle,
            "json_response",
            return_value=(200, {}, health),
        ):
            with self.assertRaisesRegex(lifecycle.LifecycleError, "HTTP 200"):
                lifecycle.maintenance_health(1234)

    def test_metric_records_are_closed_sorted_and_finite(self) -> None:
        values = {name: index for index, name in enumerate(lifecycle.METRIC_DEFINITIONS)}
        records = lifecycle.metric_records(values)
        self.assertEqual(
            [record["name"] for record in records],
            sorted(lifecycle.METRIC_DEFINITIONS),
        )
        values["adapter_load_ms"] = float("nan")
        with self.assertRaisesRegex(lifecycle.LifecycleError, "not finite"):
            lifecycle.metric_records(values)

    def test_execute_builds_once_and_reuses_the_binary_for_both_arms(self) -> None:
        digest = "sha256:" + "a" * 64
        binary = ROOT / "target" / "release" / "kiln"
        adapter = lifecycle.AdapterArm(
            config_sha256=digest,
            weights_sha256=digest,
            weights_bytes=128,
            content_revision=digest,
            generated_config_sha256=digest,
            base_before_sha256=digest,
            adapter_output_sha256=digest,
            base_after_sha256=digest,
            load_ms=1.0,
            unload_ms=2.0,
            graph_invalidation_evictions=1,
            transition_count=3,
            device_fault_count=0,
            reload_ms=3.0,
            overlap_active_adapter_ms=None,
            overlap_queued_adapter_ms=10.0,
            overlap_queued_actor_queue_ms=11.0,
            overlap_revision_header_matches=2,
        )
        maintenance = lifecycle.MaintenanceArm(
            digest,
            8,
            lifecycle.FORCED_KV_BLOCKS,
            1024,
            1.0,
            2.0,
            503,
            "maintenance_in_progress",
            0,
        )
        with (
            mock.patch.object(
                lifecycle.mixed,
                "build_binary",
                return_value=(binary, digest, 0.5),
            ) as build,
            mock.patch.object(
                lifecycle, "run_adapter_arm", return_value=adapter
            ) as adapter_arm,
            mock.patch.object(
                lifecycle, "run_maintenance_arm", return_value=maintenance
            ) as maintenance_arm,
        ):
            evidence = lifecycle.RunEvidence()
            lifecycle.execute(Path("/model"), Path("/adapter"), 17, evidence)

        build.assert_called_once()
        self.assertIs(adapter_arm.call_args.args[0], binary)
        self.assertIs(maintenance_arm.call_args.args[0], binary)
        self.assertEqual(adapter_arm.call_args.args[-2], maintenance_arm.call_args.args[-2])
        self.assertIs(adapter_arm.call_args.args[-1], evidence)
        self.assertIs(maintenance_arm.call_args.args[-1], evidence)
        metrics = lifecycle.metric_records(evidence.values)
        details = evidence.serialized_details()
        by_name = {record["name"]: record["value"] for record in metrics}
        self.assertEqual(by_name["binary_build_count"], 1)
        self.assertEqual(by_name["adapter_reload_count"], 1)
        self.assertEqual(by_name["adapter_overlap_active_adapter_phase_count"], 0)
        self.assertEqual(by_name["adapter_overlap_queued_adapter_phase_count"], 1)
        self.assertEqual(by_name["forced_resize_actual_blocks"], 1)
        self.assertEqual(json.loads(details)["kiln_binary"], digest)

    def test_partial_failure_metrics_do_not_invent_build_request_or_shutdown(self) -> None:
        evidence = lifecycle.RunEvidence()
        evidence.add_metric("binary_build_count")
        evidence.set_metric("adapter_weights_bytes", 128)
        evidence.set_metric("adapter_load_count", 1)
        evidence.set_metric("adapter_load_ms", 2.5)
        evidence.add_metric("adapter_revision_header_mismatch_count")
        evidence.arms_started.append("adapter")

        by_name = {
            record["name"]: record["value"]
            for record in lifecycle.metric_records(evidence.values)
        }
        self.assertEqual(by_name["binary_build_count"], 1)
        self.assertEqual(by_name["adapter_load_count"], 1)
        self.assertEqual(by_name["adapter_revision_header_mismatch_count"], 1)
        self.assertEqual(by_name["request_failure_count"], 0)
        self.assertEqual(by_name["dirty_shutdown_count"], 0)
        self.assertEqual(
            json.loads(evidence.serialized_details("synthetic failure"))["arms_started"],
            ["adapter"],
        )

    def test_main_serializes_partial_failure_evidence(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            model = root / "model"
            adapter = root / "adapter"
            result_path = root / "result.json"
            model.mkdir()
            adapter.mkdir()

            def fail(
                _model: Path,
                _adapter: Path,
                _seed: int,
                evidence: lifecycle.RunEvidence,
            ) -> None:
                evidence.add_metric("binary_build_count")
                evidence.set_metric("adapter_load_count", 1)
                evidence.add_metric("adapter_revision_header_mismatch_count")
                evidence.arms_started.append("adapter")
                raise lifecycle.LifecycleError("synthetic mismatch")

            with (
                mock.patch.dict(
                    lifecycle.os.environ,
                    {
                        lifecycle.RESULT_ENV: str(result_path),
                        lifecycle.VARIANT_ENV: lifecycle.VARIANT_ID,
                    },
                ),
                mock.patch.object(lifecycle, "execute", side_effect=fail),
            ):
                exit_code = lifecycle.main(
                    [
                        "--model-path",
                        str(model),
                        "--adapter-path",
                        str(adapter),
                        "--seed",
                        "17",
                    ]
                )

            self.assertEqual(exit_code, 1)
            result = json.loads(result_path.read_text())
            by_name = {
                record["name"]: record["value"] for record in result["metrics"]
            }
            self.assertEqual(by_name["binary_build_count"], 1)
            self.assertEqual(by_name["adapter_load_count"], 1)
            self.assertEqual(by_name["adapter_revision_header_mismatch_count"], 1)
            self.assertEqual(by_name["request_failure_count"], 0)
            self.assertEqual(by_name["dirty_shutdown_count"], 0)
            details = json.loads(result["details"])
            self.assertEqual(details["arms_started"], ["adapter"])
            self.assertIn("synthetic mismatch", details["error"])


if __name__ == "__main__":
    unittest.main()
