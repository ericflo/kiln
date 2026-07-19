from __future__ import annotations

import copy
import json
import sys
import tempfile
import types
import unittest
from pathlib import Path


QUALIFICATION_DIR = Path(__file__).resolve().parents[1]
ROOT = Path(__file__).resolve().parents[3]
SCRIPTS_DIR = ROOT / "scripts"
if str(QUALIFICATION_DIR) not in sys.path:
    sys.path.insert(0, str(QUALIFICATION_DIR))
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import hf_next_token_contract as contract
import json_schema_subset as schema_subset
import rocm_hf_layer_attribution as attribution
import rocm_hf_next_token_oracle as hf_oracle


REQUEST_PATH = (
    ROOT
    / "qualification/oracles/rocm-strix-halo-greedy-c1-first-divergence-v1.json"
)
ORACLE_RESULT_PATH = (
    ROOT
    / "qualification/oracle-results/rocm/strix-halo/"
    "20260719t003452-rocm-strix-halo-hf-next-token-first-divergence-v1.json"
)


def hf_evidence() -> dict[str, object]:
    evidence = copy.deepcopy(hf_oracle.validate_result(ORACLE_RESULT_PATH)["oracle"])
    evidence.update(
        {
            "boundary_count": 34,
            "boundary_names": attribution._expected_boundary_names(),
            "hidden_size": 2560,
            "layer_last_rows_sha256": "sha256:" + "8" * 64,
        }
    )
    evidence["output_bytes"] += 34 * 2560 * 4
    return evidence


def worker_marker() -> dict[str, object]:
    names = attribution._expected_boundary_names()
    relative_errors = [0.0, 0.001, 0.002, 0.025] + [0.026 + i * 0.001 for i in range(30)]
    boundaries = []
    for index, (name, relative) in enumerate(zip(names, relative_errors)):
        boundaries.append(
            {
                "cosine_similarity": 1.0 - min(relative, 0.1),
                "hf_sha256": f"sha256:{index + 1:064x}",
                "index": index,
                "kiln_sha256": f"sha256:{index + 101:064x}",
                "max_abs_error": relative * 2,
                "mean_abs_error": relative,
                "name": name,
                "reference_root_mean_square": 1.0,
                "relative_root_mean_square_error": relative,
                "root_mean_square_error": relative,
            }
        )
    return {
        "boundaries": boundaries,
        "containment": {
            "memory_current_bytes": 9_000_000_000,
            "memory_high_events": 0,
            "memory_max_bytes": 48 * 1024**3,
            "memory_max_events": 0,
            "memory_oom_events": 0,
            "memory_oom_kill_events": 0,
            "memory_peak_bytes": 14_000_000_000,
            "memory_swap_bytes": 0,
            "memory_swap_max_bytes": 0,
        },
        "final_logits_sha256": "sha256:" + "7" * 64,
        "hf_layer_last_rows_sha256": "sha256:" + "8" * 64,
        "input_token_count": 166,
        "input_token_ids_sha256": (
            "sha256:709d0a314cde9072ac79b0752e795f1b76bfaea5b553ccdf26f7fbd5ac44b1a0"
        ),
        "kernel_policy": "qualified",
        "largest_relative_error_growth": {
            "index": 3,
            "name": names[3],
            "relative_root_mean_square_error_delta": 0.023,
        },
        "observed_next_tokens": [1206, 5517, 264, 25045],
        "request_id": "rocm-strix-halo-greedy-c1-first-divergence-v1",
        "schema": attribution.WORKER_SCHEMA,
    }


class RocmHfLayerAttributionTests(unittest.TestCase):
    def test_boundary_inventory_matches_qwen35_hybrid_layout(self) -> None:
        names = attribution._expected_boundary_names()
        self.assertEqual(len(names), 34)
        self.assertEqual(names[:5], [
            "embedding",
            "layer_00_linear_attention",
            "layer_01_linear_attention",
            "layer_02_linear_attention",
            "layer_03_full_attention",
        ])
        self.assertEqual(names[-1], "final_norm")

    def test_hf_marker_extends_the_closed_next_token_evidence(self) -> None:
        evidence = hf_evidence()
        output = attribution.hf_worker.LAYER_PASS_PREFIX + json.dumps(evidence)
        self.assertEqual(attribution._parse_hf_marker(output), evidence)
        changed = copy.deepcopy(evidence)
        changed["boundary_names"][3] = "wrong"
        with self.assertRaisesRegex(attribution.LayerAttributionError, "identity"):
            attribution._parse_hf_marker(
                attribution.hf_worker.LAYER_PASS_PREFIX + json.dumps(changed)
            )

    def test_hf_worker_exposes_explicit_layer_capture_only(self) -> None:
        args = attribution.hf_worker._parse_args(
            [
                "--model",
                "/model",
                "--output",
                "/output.safetensors",
                "--request",
                "/request.json",
                "--capture-layer-last-rows",
            ]
        )
        self.assertTrue(args.capture_layer_last_rows)

    def test_hf_worker_requires_the_pinned_text_only_wrapper_depth(self) -> None:
        text_type = type("Qwen3_5TextModel", (), {})
        model_type = type("Qwen3_5ForCausalLM", (), {})
        text_model = text_type()
        text_model.config = types.SimpleNamespace(
            layer_types=[
                "full_attention" if (index + 1) % 4 == 0 else "linear_attention"
                for index in range(32)
            ]
        )
        text_model.layers = [object() for _ in range(32)]
        text_model.embed_tokens = object()
        text_model.norm = object()
        model = model_type()
        model.model = text_model

        names, modules = attribution.hf_worker._layer_capture_modules(model)
        self.assertEqual(names, attribution._expected_boundary_names())
        self.assertEqual(len(modules), 34)

        conditional_type = type("Qwen3_5ForConditionalGeneration", (), {})
        with self.assertRaisesRegex(attribution.hf_worker.OracleError, "CausalLM"):
            attribution.hf_worker._layer_capture_modules(conditional_type())

    def test_worker_recomputes_largest_relative_error_growth(self) -> None:
        marker = worker_marker()
        self.assertEqual(attribution.validate_worker_marker(marker), marker)
        changed = copy.deepcopy(marker)
        changed["largest_relative_error_growth"]["index"] = 4
        with self.assertRaisesRegex(attribution.LayerAttributionError, "growth"):
            attribution.validate_worker_marker(changed)

    def test_both_services_are_private_zero_swap_and_layer_mode_is_exact(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            workspace = Path(directory)
            hf_workspace = workspace / "hf"
            hf_workspace.mkdir()
            hf_command = attribution._bounded_hf_command(
                unit="kiln-hf-test.service",
                python=Path("/venv/python"),
                model=Path("/model"),
                request=Path("/request.json"),
                output=Path("/output.safetensors"),
                policy=Path("/policy.json"),
                workspace=hf_workspace,
            )
            self.assertIn("MemoryMax=16G", hf_command)
            self.assertIn("MemorySwapMax=0", hf_command)
            self.assertIn("PrivateNetwork=yes", hf_command)
            self.assertEqual(hf_command[-1], "--capture-layer-last-rows")

            binary = workspace / "worker"
            binary.write_bytes(b"binary")
            model = workspace / "model"
            model.mkdir()
            request = workspace / "request.json"
            reference = workspace / "reference.safetensors"
            request.write_text("{}", encoding="ascii")
            reference.write_bytes(b"reference")
            run_workspace = workspace / "run"
            run_workspace.mkdir()
            spec_path = attribution._layer_worker_spec(
                workspace=run_workspace,
                binary=binary,
                binary_sha256="sha256:" + "1" * 64,
                model=model,
                request=request,
                reference=reference,
                kernel_profile="portable_fallback",
            )
            spec = json.loads(spec_path.read_text())
            self.assertEqual(spec["argv"][1], "--layer-attribution")
            self.assertEqual(spec["argv"][-2:], ["--kernel-profile", "portable_fallback"])
            self.assertEqual(len(spec["environment"]), 6)
            self.assertFalse(any(name.startswith("KILN_") for name in spec["environment"]))

            for profile in (
                "fused_norm_mlp_fallback",
                "fused_norm_mlp_only",
                "model_fallback",
                "tensor_fallback",
                "gdn_fallback",
                "non_gdn_fallback",
                "split_q_gate_fallback",
                "split_q_gate_only",
            ):
                profile_workspace = workspace / profile
                profile_workspace.mkdir()
                profile_spec = attribution._layer_worker_spec(
                    workspace=profile_workspace,
                    binary=binary,
                    binary_sha256="sha256:" + "1" * 64,
                    model=model,
                    request=request,
                    reference=reference,
                    kernel_profile=profile,
                )
                profile_document = json.loads(profile_spec.read_text())
                self.assertEqual(
                    profile_document["argv"][-2:], ["--kernel-profile", profile]
                )

            with self.assertRaisesRegex(attribution.LayerAttributionError, "unsupported"):
                attribution._layer_worker_spec(
                    workspace=workspace / "invalid",
                    binary=binary,
                    binary_sha256="sha256:" + "1" * 64,
                    model=model,
                    request=request,
                    reference=reference,
                    kernel_profile="individual_switches",
                )

    def test_result_schema_and_checker_bind_request_reference_and_growth(self) -> None:
        request, request_sha256 = contract.load_request(REQUEST_PATH)
        oracle = hf_oracle.validate_result(ORACLE_RESULT_PATH)
        evidence = hf_evidence()
        result = {
            "binary": {
                "build_duration_seconds": 1.0,
                "build_environment_policy": "closed-source-build-v1",
                "path": "target/release/examples/rocm_hf_path_attribution",
                "rocm_archs": ["gfx1151"],
                "sha256": "sha256:" + "1" * 64,
            },
            "containment": {
                "hf": {
                    "host_available_before_gib": 25,
                    "memory_max_gib": 16,
                    "network": "forbidden",
                    "thermal": oracle["containment"]["service"],
                },
                "kiln": {
                    "host_available_before_gib": 25,
                    "memory_max_gib": 48,
                    "network": "forbidden",
                    "thermal": oracle["containment"]["service"],
                },
            },
            "created_at_utc": "2026-07-19T02:00:00Z",
            "duration_seconds": 60.0,
            "implementation": {
                "guarded_exec_sha256": "sha256:" + "2" * 64,
                "hf_worker_sha256": "sha256:" + "3" * 64,
                "python_sha256": "sha256:" + "4" * 64,
                "runner_sha256": "sha256:" + "5" * 64,
                "supervisor_sha256": "sha256:" + "6" * 64,
            },
            "model_fingerprint": {
                "implementation_sha256": "sha256:" + "7" * 64,
                "python_sha256": "sha256:" + "4" * 64,
                "thermal": oracle["containment"]["service"],
            },
            "model_identity": request["model_identity"],
            "reference": {
                "bytes": evidence["output_bytes"],
                "evidence": evidence,
                "location": "local_ignored",
                "sha256": "sha256:" + "9" * 64,
            },
            "request": {
                "path": REQUEST_PATH.relative_to(ROOT).as_posix(),
                "sha256": request_sha256,
            },
            "schema": attribution.SCHEMA,
            "source": {"commit": "a" * 40, "origin_main": "a" * 40, "tree": "b" * 40},
            "worker": worker_marker(),
        }
        result["result_sha256"] = contract.canonical_sha256(result)
        schema = json.loads(
            (ROOT / "qualification/schema/rocm-hf-layer-attribution-v1.schema.json").read_text()
        )
        oracle_schema = json.loads(
            (ROOT / "qualification/schema/rocm-hf-next-token-oracle-v1.schema.json").read_text()
        )
        self.assertEqual(
            schema_subset.validate_instance(
                result,
                schema,
                schema,
                registry={"rocm-hf-next-token-oracle-v1.schema.json": oracle_schema},
            ),
            [],
        )
        fallback = copy.deepcopy(result)
        fallback["worker"]["kernel_policy"] = "portable_fallback"
        fallback["result_sha256"] = contract.canonical_sha256(
            {name: value for name, value in fallback.items() if name != "result_sha256"}
        )
        self.assertEqual(
            schema_subset.validate_instance(
                fallback,
                schema,
                schema,
                registry={"rocm-hf-next-token-oracle-v1.schema.json": oracle_schema},
            ),
            [],
        )
        self.assertEqual(attribution.validate_worker_marker(fallback["worker"]), fallback["worker"])
        for profile in (
            "fused_norm_mlp_fallback",
            "fused_norm_mlp_only",
            "model_fallback",
            "tensor_fallback",
            "gdn_fallback",
            "non_gdn_fallback",
            "split_q_gate_fallback",
            "split_q_gate_only",
        ):
            diagnostic = copy.deepcopy(fallback["worker"])
            diagnostic["kernel_policy"] = profile
            self.assertEqual(attribution.validate_worker_marker(diagnostic), diagnostic)
        invalid = copy.deepcopy(fallback["worker"])
        invalid["kernel_policy"] = "individual_switches"
        with self.assertRaisesRegex(attribution.LayerAttributionError, "request or output"):
            attribution.validate_worker_marker(invalid)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "result.json"
            path.write_text(json.dumps(result), encoding="ascii")
            self.assertEqual(attribution.validate_result(path), result)
            result["worker"]["largest_relative_error_growth"]["index"] = 4
            path.write_text(json.dumps(result), encoding="ascii")
            with self.assertRaisesRegex(attribution.LayerAttributionError, "result_sha256"):
                attribution.validate_result(path)


if __name__ == "__main__":
    unittest.main()
