from __future__ import annotations

import copy
import json
import sys
import tempfile
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
import rocm_hf_next_token_oracle as hf_oracle
import rocm_hf_path_attribution as attribution


REQUEST_PATH = (
    ROOT
    / "qualification/oracles/rocm-strix-halo-greedy-c1-first-divergence-v1.json"
)
ORACLE_RESULT_PATH = (
    ROOT
    / "qualification/oracle-results/rocm/strix-halo/"
    "20260719t003452-rocm-strix-halo-hf-next-token-first-divergence-v1.json"
)


def comparison(argmax: int, *, matches: bool) -> dict[str, object]:
    return {
        "argmax": argmax,
        "argmax_equal": matches,
        "candidate_tokens": [
            {"logit": 18.25, "rank": 2, "token_id": 25045},
            {"logit": 18.5, "rank": 1, "token_id": 15787},
        ],
        "cosine_similarity": 0.999,
        "logits_sha256": "sha256:" + "1" * 64,
        "max_abs_error": 0.5,
        "mean_abs_error": 0.01,
        "top10_overlap": 9,
    }


def worker_marker() -> dict[str, object]:
    return {
        "attribution": "eager_full_logits",
        "containment": {
            "memory_current_bytes": 10_000_000_000,
            "memory_high_events": 0,
            "memory_max_bytes": attribution.MEMORY_MAX_GIB * 1024**3,
            "memory_max_events": 0,
            "memory_oom_events": 0,
            "memory_oom_kill_events": 0,
            "memory_peak_bytes": 12_000_000_000,
            "memory_swap_bytes": 0,
            "memory_swap_max_bytes": 0,
        },
        "eager_full": {
            "comparison": comparison(25045, matches=False),
            "observed_next_tokens": [1206, 5517, 264, 25045],
        },
        "eager_greedy": {
            "final_token_matches_reference": False,
            "observed_next_tokens": [1206, 5517, 264, 25045],
        },
        "graph": {
            "cache_admission_successes": 1,
            "capture_attempts": 2,
            "capture_failures": 0,
            "capture_successes": 1,
            "enabled": True,
            "fallbacks": 0,
            "replay_attempts": 7,
            "replay_failures": 0,
            "replay_successes": 7,
        },
        "graph_full": {
            "comparison": comparison(15787, matches=True),
            "observed_next_tokens": [1206, 5517, 264, 15787],
        },
        "graph_greedy": {
            "final_token_matches_reference": True,
            "observed_next_tokens": [1206, 5517, 264, 15787],
        },
        "hf_argmax": 15787,
        "input_token_count": 166,
        "input_token_ids_sha256": (
            "sha256:709d0a314cde9072ac79b0752e795f1b76bfaea5b553ccdf26f7fbd5ac44b1a0"
        ),
        "kernel_policy": "qualified",
        "request_id": "rocm-strix-halo-greedy-c1-first-divergence-v1",
        "schema": attribution.WORKER_SCHEMA,
    }


class RocmHfPathAttributionTests(unittest.TestCase):
    def test_build_environment_is_closed_and_source_bound(self) -> None:
        environment = attribution._build_environment(
            {
                "HOME": "/home/test",
                "KILN_AMBIENT_PRODUCT_CONTROL": "must-not-enter",
                "RUSTFLAGS": "must-be-filtered-by-the-closed-service",
            }
        )
        self.assertNotIn("KILN_AMBIENT_PRODUCT_CONTROL", environment)
        self.assertEqual(environment["KILN_ROCM_ARCHS"], "gfx1151")
        self.assertEqual(
            environment["KILN_CARGO_ENVIRONMENT_POLICY"], "closed-source-build-v1"
        )
        self.assertEqual(environment["KILN_CARGO_EXECUTION_MODE"], "transient-service")
        self.assertEqual(environment["KILN_CARGO_PRIVATE_NETWORK"], "1")
        self.assertEqual(environment["RUSTFLAGS"], "must-be-filtered-by-the-closed-service")

    def test_worker_marker_is_closed_and_recomputes_attribution(self) -> None:
        marker = worker_marker()
        self.assertEqual(attribution.validate_worker_marker(marker), marker)

        changed = copy.deepcopy(marker)
        changed["graph"]["replay_successes"] = 6
        with self.assertRaisesRegex(attribution.AttributionError, "retained replay"):
            attribution.validate_worker_marker(changed)
        changed = copy.deepcopy(marker)
        changed["attribution"] = "hip_graph_full_logits"
        with self.assertRaisesRegex(attribution.AttributionError, "attribution"):
            attribution.validate_worker_marker(changed)

    def test_service_is_private_zero_swap_and_runs_hash_bound_shim(self) -> None:
        command = attribution._service_command(
            python=Path("/venv/bin/python"),
            policy=Path("/repo/policy.json"),
            workspace=Path("/run/workspace"),
            spec=Path("/run/workspace/spec.json"),
            unit="kiln-test.service",
        )
        for expected in (
            "MemoryMax=48G",
            "MemorySwapMax=0",
            "KillMode=control-group",
            "PrivateNetwork=yes",
            str(attribution.SUPERVISOR),
            str(attribution.GUARDED_EXEC),
            "--spec",
        ):
            self.assertIn(expected, command)
        self.assertNotIn("KILN_", "\0".join(command))

    def test_result_checker_binds_retained_oracle_request_and_self_hash(self) -> None:
        request, request_sha256 = contract.load_request(REQUEST_PATH)
        oracle = hf_oracle.validate_result(ORACLE_RESULT_PATH)
        marker = worker_marker()
        result = {
            "binary": {
                "build_duration_seconds": 1.0,
                "build_environment_policy": attribution.BUILD_ENVIRONMENT_POLICY,
                "path": "target/release/examples/rocm_hf_path_attribution",
                "rocm_archs": [attribution.BUILD_ROCM_ARCHS],
                "sha256": "sha256:" + "2" * 64,
            },
            "containment": {
                "host_available_before_gib": 25,
                "memory_max_gib": attribution.MEMORY_MAX_GIB,
                "network": "forbidden",
                "thermal": oracle["containment"]["service"],
            },
            "created_at_utc": "2026-07-19T01:00:00Z",
            "duration_seconds": 30.0,
            "implementation": {
                "guarded_exec_sha256": "sha256:" + "3" * 64,
                "runner_sha256": "sha256:" + "4" * 64,
                "supervisor_sha256": "sha256:" + "5" * 64,
            },
            "model_fingerprint": {
                "implementation_sha256": "sha256:" + "6" * 64,
                "python_sha256": "sha256:" + "7" * 64,
                "thermal": oracle["containment"]["service"],
            },
            "model_identity": request["model_identity"],
            "oracle_reference": {
                "bytes": oracle["reference_artifact"]["bytes"],
                "oracle_result_path": ORACLE_RESULT_PATH.relative_to(ROOT).as_posix(),
                "oracle_result_sha256": oracle["result_sha256"],
                "raw_location": "local_ignored",
                "raw_sha256": oracle["reference_artifact"]["sha256"],
            },
            "request": {
                "path": REQUEST_PATH.relative_to(ROOT).as_posix(),
                "sha256": request_sha256,
            },
            "schema": attribution.SCHEMA,
            "source": {
                "commit": "a" * 40,
                "origin_main": "a" * 40,
                "tree": "b" * 40,
            },
            "worker": marker,
        }
        result["result_sha256"] = contract.canonical_sha256(result)
        schema = json.loads(
            (
                ROOT
                / "qualification/schema/rocm-hf-path-attribution-v1.schema.json"
            ).read_text()
        )
        oracle_schema = json.loads(
            (
                ROOT
                / "qualification/schema/rocm-hf-next-token-oracle-v1.schema.json"
            ).read_text()
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
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "result.json"
            path.write_text(json.dumps(result), encoding="ascii")
            self.assertEqual(attribution.validate_result(path), result)
            result["worker"]["attribution"] = "hip_graph_full_logits"
            path.write_text(json.dumps(result), encoding="ascii")
            with self.assertRaisesRegex(attribution.AttributionError, "result_sha256"):
                attribution.validate_result(path)


if __name__ == "__main__":
    unittest.main()
