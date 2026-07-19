from __future__ import annotations

import copy
import importlib.util
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock


QUALIFICATION_DIR = Path(__file__).resolve().parents[1]
ROOT = Path(__file__).resolve().parents[3]
if str(QUALIFICATION_DIR) not in sys.path:
    sys.path.insert(0, str(QUALIFICATION_DIR))

import hf_next_token_contract as contract
import hf_thermal_supervisor as supervisor
import host_thermal_policy
import rocm_hf_next_token_oracle as runner

SPEC = importlib.util.spec_from_file_location(
    "qwen35_hf_logits_for_test", QUALIFICATION_DIR / "qwen35_hf_logits.py"
)
assert SPEC is not None and SPEC.loader is not None
worker = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = worker
SPEC.loader.exec_module(worker)

SCHEMA_SPEC = importlib.util.spec_from_file_location(
    "json_schema_subset_for_hf_oracle_test", ROOT / "scripts/json_schema_subset.py"
)
assert SCHEMA_SPEC is not None and SCHEMA_SPEC.loader is not None
schema_subset = importlib.util.module_from_spec(SCHEMA_SPEC)
sys.modules[SCHEMA_SPEC.name] = schema_subset
SCHEMA_SPEC.loader.exec_module(schema_subset)


REQUEST_PATH = (
    ROOT
    / "qualification/oracles/rocm-strix-halo-greedy-c1-first-divergence-v1.json"
)
POLICY_PATH = ROOT / "qualification/host-policies/strix-halo-hf-oracle-v1.json"
ALTERNATE_POLICY_PATH = (
    ROOT
    / "qualification/host-policies/strix-halo-serving-benchmark-hard-limit-v1.json"
)


def next_token_evidence() -> dict[str, object]:
    top = [
        {"logit": 10.0 - index, "text": f"token-{index}", "token_id": 25045 + index}
        for index in range(10)
    ]
    top[0]["text"] = " baseline"
    top[1]["text"] = " foundation"
    top[1]["token_id"] = 15787
    return {
        "attention_implementation": "eager",
        "argmax": 25045,
        "argmax_text": " baseline",
        "candidate_tokens": [
            {
                "engine": "kiln",
                "logit": 10.0,
                "rank": 1,
                "text": " baseline",
                "token_id": 25045,
            },
            {
                "engine": "vllm",
                "logit": 9.0,
                "rank": 2,
                "text": " foundation",
                "token_id": 15787,
            },
        ],
        "configuration_sha256": "sha256:3c01b3cdcff8d77cbafac9841bc48c41e5a5b38637231f1bde3d843cd198dbaf",
        "deterministic_algorithms": True,
        "device": "AMD Radeon 8060S Graphics",
        "dtype": "bfloat16",
        "duration_seconds": 5.0,
        "input_token_count": 166,
        "input_token_ids_sha256": "sha256:" + "a" * 64,
        "logits_sha256": "sha256:" + "b" * 64,
        "linear_attention_implementation": "transformers_torch_fallback",
        "memory_high_events": 0,
        "memory_max_events": 0,
        "memory_oom_events": 0,
        "memory_oom_kill_events": 0,
        "memory_peak_bytes": 10_000_000_000,
        "memory_swap_bytes": 0,
        "modeling_sha256": "sha256:cf085792cb59e5bdf9b88a3d20bd353892289d054662a9c2b662221b97caefba",
        "output_bytes": 994_000,
        "request_id": "request-v1",
        "request_sha256": "sha256:" + "c" * 64,
        "tf32_allowed": False,
        "top_logit_margin": 1.0,
        "top_tokens": top,
        "torch_hip_version": "7.2.53211",
        "torch_commit": "cf30153c4c131c8164ee7798e5022d810682e2cb",
        "torch_version": "2.13.0+rocm7.2",
        "transformers_version": "5.13.1",
        "vocab": 248_320,
    }


def thermal_evidence(policy_path: Path = POLICY_PATH) -> dict[str, object]:
    policy, runtime_policy, settlement = host_thermal_policy.load(
        policy_path,
        cooldown_mode="post_process_exit_consecutive_samples",
    )
    safe_temperature = runtime_policy.cooldown_target_millicelsius - 1_000
    peak_temperature = min(80_000, runtime_policy.limit_millicelsius - 1_000)
    return {
        "phase_settlement_timeout_seconds": settlement,
        "policy": policy,
        "prelaunch_cooldown": {
            "completed": True,
            "elapsed_seconds": 1.0,
            "poll_interval_ms": runtime_policy.poll_interval_ms,
            "sample_count": runtime_policy.cooldown_stable_samples,
            "scope": "host_package_before_process_creation",
            "sensor_path": "/sys/class/hwmon/hwmon0/temp1_input",
            "stable_samples_observed": runtime_policy.cooldown_stable_samples,
            "stable_samples_required": runtime_policy.cooldown_stable_samples,
            "target_millicelsius": runtime_policy.cooldown_target_millicelsius,
            "temperature_end_millicelsius": safe_temperature,
            "temperature_peak_millicelsius": runtime_policy.cooldown_target_millicelsius,
            "temperature_start_millicelsius": runtime_policy.cooldown_target_millicelsius,
            "timeout_seconds": runtime_policy.cooldown_timeout_seconds,
        },
        "runtime": {
            "host_temperature_end_millicelsius": safe_temperature,
            "host_temperature_peak_millicelsius": peak_temperature,
            "host_temperature_start_millicelsius": safe_temperature,
            "host_thermal_cooldown_active_end": 0,
            "host_thermal_cooldown_completed_count": 1,
            "host_thermal_cooldown_peak_millicelsius": peak_temperature,
            "host_thermal_cooldown_sample_count": 100,
            "host_thermal_cooldown_seconds": 5.0,
            "host_thermal_cooldown_stable_sample_count": runtime_policy.cooldown_stable_samples,
            "host_thermal_cooldown_timeout_count": 0,
            "host_thermal_guard_trip_count": 0,
            "host_thermal_pacing_active_end": 0,
            "host_thermal_pacing_completed_event_count": 2,
            "host_thermal_pacing_event_count": 2,
            "host_thermal_pacing_max_seconds": 2.0,
            "host_thermal_pacing_max_start_millicelsius": (
                runtime_policy.pacing_start_millicelsius or 0
            ),
            "host_thermal_pacing_seconds": 3.0,
        },
        "schema": supervisor.SCHEMA,
        "worker_exit_code": 0,
    }


class HfNextTokenOracleTests(unittest.TestCase):
    def test_tracked_request_is_closed_source_bound_and_exact(self) -> None:
        request, digest = contract.load_request(REQUEST_PATH)
        contract.validate_source_receipts(request, ROOT)
        self.assertEqual(
            digest,
            "sha256:fbe61941ebc147d50141a04bb8541e7ccb5bc16d6c893285ec9a2660f51336c4",
        )
        self.assertEqual(len(request["prompt"]["token_ids"]), 163)
        self.assertEqual(len(request["input_token_ids"]), 166)
        self.assertEqual(
            [item["token_id"] for item in request["continuation_prefix"]],
            [1206, 5517, 264],
        )
        self.assertEqual(
            [item["token_id"] for item in request["candidates"]],
            [25045, 15787],
        )
        schema = json.loads(
            (
                ROOT
                / "qualification/schema/hf-next-token-request-v1.schema.json"
            ).read_text()
        )
        self.assertEqual(schema_subset.validate_instance(request, schema, schema), [])

    def test_request_rejects_unknown_fields_and_inconsistent_input(self) -> None:
        request = json.loads(REQUEST_PATH.read_text())
        unknown = copy.deepcopy(request)
        unknown["unexpected"] = True
        with self.assertRaisesRegex(contract.ContractError, "not closed"):
            contract.validate_request(unknown)
        changed = copy.deepcopy(request)
        changed["input_token_ids"][-1] = 1
        with self.assertRaisesRegex(contract.ContractError, "prompt plus continuation"):
            contract.validate_request(changed)

    def test_next_token_marker_is_unique_and_closed(self) -> None:
        evidence = next_token_evidence()
        marker = contract.PASS_PREFIX + json.dumps(evidence)
        self.assertEqual(contract.parse_pass_marker(marker), evidence)
        with self.assertRaisesRegex(contract.ContractError, "found 2"):
            contract.parse_pass_marker(marker + "\n" + marker)
        evidence["unexpected"] = 1
        with self.assertRaisesRegex(contract.ContractError, "not closed"):
            contract.parse_pass_marker(contract.PASS_PREFIX + json.dumps(evidence))

    def test_worker_gate_requires_absolute_exact_regular_release(self) -> None:
        with self.assertRaisesRegex(worker.OracleError, "must be absolute"):
            worker._wait_for_start_gate(Path("relative"), timeout_seconds=0.01)
        with tempfile.TemporaryDirectory() as directory:
            gate = Path(directory) / "gate"
            gate.write_bytes(b"go\n")
            worker._wait_for_start_gate(gate, timeout_seconds=0.01)
            gate.write_bytes(b"wrong")
            with self.assertRaisesRegex(worker.OracleError, "payload"):
                worker._wait_for_start_gate(gate, timeout_seconds=0.01)

    def test_tracked_policy_is_closed_and_content_hashed(self) -> None:
        record, policy, settlement = host_thermal_policy.load(POLICY_PATH)
        self.assertEqual(
            record["content_sha256"],
            "sha256:3c175cfbf85da62e63ff4c8f01facdcd679c066d85c0a9595451aa26260f87c3",
        )
        self.assertEqual(policy.cooldown_mode, "live_process_safe_handoff")
        self.assertEqual(settlement, 300.0)

    def test_systemd_command_is_private_bounded_and_runs_supervisor(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            workspace = Path(directory)
            command = runner._bounded_command(
                unit="kiln-rocm-hf-next-token-test.service",
                python=Path("/venv/bin/python"),
                model=Path("/models/qwen"),
                request=Path("/repo/request.json"),
                output=Path("/run/reference.safetensors"),
                policy=Path("/repo/policy.json"),
                workspace=workspace,
            )
        for expected in (
            "MemoryMax=16G",
            "MemorySwapMax=0",
            "KillMode=control-group",
            "PrivateNetwork=yes",
            "/usr/bin/env",
            "-i",
            "HF_HUB_OFFLINE=1",
            "TRANSFORMERS_OFFLINE=1",
            str(runner.SUPERVISOR_SCRIPT),
            "--host-thermal-policy",
            str(runner.HF_SCRIPT),
            "--request",
        ):
            self.assertIn(expected, command)
        joined = "\0".join(command)
        self.assertNotIn("HF_TOKEN", joined)
        self.assertNotIn("GITHUB_TOKEN", joined)

    def test_supervisor_marker_requires_clean_cooldown(self) -> None:
        evidence = thermal_evidence()
        marker = supervisor.PASS_PREFIX + json.dumps(evidence)
        self.assertEqual(supervisor.parse_pass_marker(marker), evidence)
        evidence["runtime"]["host_thermal_guard_trip_count"] = 1
        with self.assertRaisesRegex(supervisor.SupervisorError, "guard trip"):
            supervisor.parse_pass_marker(supervisor.PASS_PREFIX + json.dumps(evidence))

    def test_supervisor_worker_command_owns_start_gate(self) -> None:
        command = supervisor._validate_worker_command(
            [sys.executable, str(REQUEST_PATH)]
        )
        self.assertEqual(command[0], sys.executable)
        with self.assertRaisesRegex(supervisor.SupervisorError, "own --start-gate"):
            supervisor._validate_worker_command(
                [sys.executable, str(REQUEST_PATH), "--start-gate", "/tmp/gate"]
            )

    def test_model_fingerprint_runs_as_guarded_worker_with_closed_environment(self) -> None:
        request, _request_sha256 = contract.load_request(REQUEST_PATH)
        raw_identity = {**request["model_identity"], "path": "/models/pinned"}
        raw_identity.pop("content_sha256")
        stdout = json.dumps(raw_identity)
        with tempfile.TemporaryDirectory() as directory, mock.patch.object(
            supervisor,
            "supervise",
            return_value=(0, stdout, "", thermal_evidence()),
        ) as supervise:
            model = Path(directory) / "model"
            model.mkdir()
            actual_model, identity, evidence = runner._validate_model(
                model,
                request["model_identity"],
                policy=POLICY_PATH,
                python=Path(sys.executable).resolve(),
            )
        command = supervise.call_args.kwargs["worker_command"]
        environment = supervise.call_args.kwargs["worker_environment"]
        self.assertEqual(actual_model, model.resolve())
        self.assertEqual(identity, request["model_identity"])
        self.assertEqual(evidence["thermal"], thermal_evidence())
        self.assertEqual(
            set(environment), {"HOME", "LANG", "LC_ALL", "PATH", "PYTHONHASHSEED", "TMPDIR"}
        )
        self.assertEqual(command[1], str(runner.MODEL_FINGERPRINT_SCRIPT))
        self.assertNotIn("--start-gate", command)
        self.assertEqual(supervise.call_args.kwargs["worker_phase"], "model-fingerprint")
        self.assertNotIn("HF_TOKEN", environment)
        self.assertNotIn("KILN_", "\0".join(f"{key}={value}" for key, value in environment.items()))

    def test_guarded_fingerprint_output_requires_strict_json_object(self) -> None:
        identity = runner._parse_guarded_fingerprint_output('{"id":"model"}')
        self.assertEqual(identity, {"id": "model"})
        with self.assertRaisesRegex(runner.OracleRunError, "invalid JSON"):
            runner._parse_guarded_fingerprint_output("noise")
        with self.assertRaisesRegex(runner.OracleRunError, "must be an object"):
            runner._parse_guarded_fingerprint_output("[]")

    def test_result_checker_binds_request_containment_and_self_hash(self) -> None:
        request, request_sha256 = contract.load_request(REQUEST_PATH)
        oracle = next_token_evidence()
        oracle["request_id"] = request["id"]
        oracle["request_sha256"] = request_sha256
        oracle["input_token_ids_sha256"] = request["input_token_ids_sha256"]
        result = {
            "containment": {
                "host_available_before_gib": 24,
                "memory_max_gib": 16,
                "network": "forbidden",
                "service": thermal_evidence(),
                "swap_max_bytes": 0,
            },
            "created_at_utc": "2026-07-18T00:00:00Z",
            "duration_seconds": 10.0,
            "implementation": {
                "hf_worker_sha256": "sha256:" + "1" * 64,
                "python_sha256": "sha256:" + "2" * 64,
                "supervisor_sha256": "sha256:" + "3" * 64,
            },
            "model_fingerprint": {
                "implementation_sha256": "sha256:" + "5" * 64,
                "python_sha256": "sha256:" + "2" * 64,
                "thermal": thermal_evidence(),
            },
            "model_identity": request["model_identity"],
            "oracle": oracle,
            "reference_artifact": {
                "bytes": oracle["output_bytes"],
                "location": "local_ignored",
                "sha256": "sha256:" + "4" * 64,
            },
            "request": {
                "contract_path": REQUEST_PATH.relative_to(ROOT).as_posix(),
                "id": request["id"],
                "sha256": request_sha256,
                "source": request["source"],
            },
            "schema": runner.SCHEMA,
            "source": {
                "commit": "a" * 40,
                "origin_main": "a" * 40,
                "tree": "b" * 40,
            },
            "verdict": {
                "argmax_candidate": "kiln",
                "argmax_token_id": 25045,
                "candidate_attribution_complete": True,
            },
        }
        result["result_sha256"] = contract.canonical_sha256(result)
        schema = json.loads(
            (
                ROOT
                / "qualification/schema/rocm-hf-next-token-oracle-v1.schema.json"
            ).read_text()
        )
        self.assertEqual(schema_subset.validate_instance(result, schema, schema), [])
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "result.json"
            path.write_text(json.dumps(result))
            self.assertEqual(runner.validate_result(path), result)
            mismatch = copy.deepcopy(result)
            mismatch["model_fingerprint"]["thermal"] = thermal_evidence(ALTERNATE_POLICY_PATH)
            mismatch["result_sha256"] = contract.canonical_sha256(
                {name: value for name, value in mismatch.items() if name != "result_sha256"}
            )
            path.write_text(json.dumps(mismatch))
            with self.assertRaisesRegex(runner.OracleRunError, "thermal policies differ"):
                runner.validate_result(path)
            missing = copy.deepcopy(result)
            missing.pop("model_fingerprint")
            missing["result_sha256"] = contract.canonical_sha256(
                {name: value for name, value in missing.items() if name != "result_sha256"}
            )
            path.write_text(json.dumps(missing))
            with self.assertRaisesRegex(runner.OracleRunError, "thermal evidence is required"):
                runner.validate_result(path)
            result["verdict"]["argmax_candidate"] = "vllm"
            path.write_text(json.dumps(result))
            with self.assertRaisesRegex(runner.OracleRunError, "result_sha256"):
                runner.validate_result(path)

    def test_check_command_validates_every_retained_result(self) -> None:
        first = Path("qualification/oracle-results/first.json")
        second = Path("qualification/oracle-results/second.json")
        validated = [
            {"result_sha256": "sha256:" + "1" * 64},
            {"result_sha256": "sha256:" + "2" * 64},
        ]
        with mock.patch.object(runner, "validate_result", side_effect=validated) as check:
            self.assertEqual(runner.main(["check", str(first), str(second)]), 0)
        self.assertEqual(
            check.call_args_list,
            [
                mock.call(first, require_current_source=False),
                mock.call(second, require_current_source=False),
            ],
        )


if __name__ == "__main__":
    unittest.main()
