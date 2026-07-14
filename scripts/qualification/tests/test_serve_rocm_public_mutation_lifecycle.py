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

    def test_driver_uses_typed_config_and_one_binary_for_both_arms(self) -> None:
        source = (
            QUALIFICATION_DIR / "serve_rocm_public_mutation_lifecycle.py"
        ).read_text()
        self.assertEqual(source.count("mixed.build_binary(deadline)"), 1)
        self.assertEqual(source.count("mixed.write_server_config("), 2)
        self.assertIn('kv_force_blocks=FORCED_KV_BLOCKS', source)
        self.assertIn('[str(binary), "--config", str(config_path), "serve"]', source)
        self.assertNotIn('"--config", "/dev/null"', source)
        self.assertNotIn("KILN_KV_FORCE_BLOCKS", source)


if __name__ == "__main__":
    unittest.main()
