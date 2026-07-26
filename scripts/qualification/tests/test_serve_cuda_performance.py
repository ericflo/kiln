import importlib.util
import json
import os
import sys
import tempfile
import unittest
from unittest import mock
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
SCRIPT = ROOT / "scripts/qualification/serve_cuda_performance.py"
sys.path.insert(0, str(SCRIPT.parent))
SPEC = importlib.util.spec_from_file_location("serve_cuda_performance", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
performance = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = performance
SPEC.loader.exec_module(performance)


class CudaPerformanceTests(unittest.TestCase):
    def test_effective_config_matches_the_committed_workload(self) -> None:
        workload = json.loads(
            (
                ROOT
                / "qualification/workloads/serving-cuda-performance-c1-v1.json"
            ).read_text()
        )
        variant = workload["variants"][0]
        self.assertEqual(variant["id"], performance.VARIANT_ID)
        self.assertEqual(
            variant["cases"][0]["id"],
            performance.CASE_ID,
        )
        self.assertEqual(variant["effective_config"], performance.EFFECTIVE_CONFIG)
        self.assertEqual(
            variant["cases"][0]["result_protocol"]["declared_metrics"],
            sorted(performance.METRIC_DEFINITIONS),
        )

    def test_tracked_inputs_match_the_closed_hashes(self) -> None:
        inputs = (
            (
                performance.KILN_LAUNCH,
                performance.KILN_LAUNCH_SHA256,
            ),
            (
                performance.VLLM_LAUNCH,
                performance.VLLM_LAUNCH_SHA256,
            ),
            (
                performance.KILN_CONFIG,
                performance.KILN_CONFIG_SHA256,
            ),
            (
                performance.VLLM_RUNTIME_MANIFEST,
                performance.VLLM_RUNTIME_MANIFEST_SHA256,
            ),
        )
        for path, expected in inputs:
            with self.subTest(path=path):
                self.assertEqual(performance.sha256_file(path), expected)

    def test_campaign_commands_preserve_the_paired_contract(self) -> None:
        model = ROOT / ".qualification/model"
        binary = ROOT / "target/release/kiln"
        commit = "a" * 40
        output = ROOT / ".qualification/performance/kiln"
        kiln = performance.campaign_command(
            engine="kiln",
            model_path=model,
            commit=commit,
            output_directory=output,
            binary=binary,
            reference_directory=None,
        )
        vllm = performance.campaign_command(
            engine="vllm",
            model_path=model,
            commit=commit,
            output_directory=ROOT / ".qualification/performance/vllm",
            binary=binary,
            reference_directory=output,
        )
        self.assertIn("1", kiln)
        self.assertIn(performance.GPU_UUID, kiln)
        self.assertIn(str(performance.THERMAL_POLICY), kiln)
        self.assertIn(str(performance.KILN_LAUNCH), kiln)
        self.assertNotIn("--reference-dir", kiln)
        self.assertIn(str(performance.VLLM_RUNTIME_MANIFEST), vllm)
        self.assertIn(str(performance.VLLM_LAUNCH), vllm)
        self.assertEqual(vllm[-2:], ["--reference-dir", str(output)])

    def test_build_uses_the_closed_delegated_cuda_command(self) -> None:
        binary = performance.ROOT / "target/release/kiln"
        process = mock.Mock()
        process.poll.return_value = 0
        process.returncode = 0
        with mock.patch.object(
            performance.subprocess,
            "Popen",
            return_value=process,
        ) as popen, mock.patch.object(
            performance,
            "_build_elapsed_seconds",
            return_value=(1.0, 0.0, 1.0),
        ), mock.patch.object(
            performance.os,
            "access",
            return_value=True,
        ), mock.patch.object(
            performance.Path,
            "is_file",
            return_value=True,
        ), mock.patch.object(
            performance.Path,
            "is_symlink",
            return_value=False,
        ), mock.patch.object(
            performance,
            "sha256_file",
            return_value="sha256:" + "a" * 64,
        ):
            observed_binary, _, active, wall, paused = performance.build_binary(
                performance.time.monotonic() + 60.0
            )
        self.assertEqual(observed_binary, binary)
        self.assertEqual((active, wall, paused), (1.0, 1.0, 0.0))
        self.assertEqual(
            popen.call_args.args[0],
            [
                str(performance.ROOT / "scripts/cargo-bounded.sh"),
                "build",
                "--locked",
                "--offline",
                "--release",
                "-p",
                "kiln-server",
                "--bin",
                "kiln",
                "--no-default-features",
                "--features",
                "cuda",
            ],
        )
        environment = popen.call_args.kwargs["env"]
        self.assertEqual(environment["CARGO_NET_OFFLINE"], "true")
        self.assertEqual(
            environment["KILN_CARGO_EXECUTION_MODE"], "delegated-cgroup"
        )
        self.assertEqual(environment["KILN_CARGO_MAX_MEMORY_GIB"], "10")
        self.assertEqual(environment["KILN_CARGO_CPU_QUOTA_PERCENT"], "50")
        self.assertNotIn("KILN_CARGO_SERVICE_RUNTIME_MAX_SECONDS", environment)
        self.assertNotEqual(environment, os.environ)
        self.assertTrue(popen.call_args.kwargs["start_new_session"])

    def test_build_rejects_active_timeout_and_terminates_group(self) -> None:
        process = mock.Mock()
        process.poll.return_value = None
        with mock.patch.object(
            performance.subprocess,
            "Popen",
            return_value=process,
        ), mock.patch.object(
            performance,
            "_build_elapsed_seconds",
            return_value=(1800.1, 0.0, 1800.1),
        ), mock.patch.object(
            performance,
            "_terminate_build_process",
        ) as terminate:
            with self.assertRaisesRegex(
                performance.mixed.QualificationError,
                "1800.000 active seconds",
            ):
                performance.build_binary(performance.time.monotonic() + 2000.0)
        terminate.assert_called_once_with(process)

    def test_build_rejects_excess_verified_thermal_pacing(self) -> None:
        process = mock.Mock()
        process.poll.return_value = None
        with mock.patch.object(
            performance.subprocess,
            "Popen",
            return_value=process,
        ), mock.patch.object(
            performance,
            "_build_elapsed_seconds",
            return_value=(14401.0, 14400.1, 0.9),
        ), mock.patch.object(
            performance,
            "_terminate_build_process",
        ) as terminate:
            with self.assertRaisesRegex(
                performance.mixed.QualificationError,
                "verified thermal pacing",
            ):
                performance.build_binary(performance.time.monotonic() + 16000.0)
        terminate.assert_called_once_with(process)

    def test_build_evidence_failure_terminates_group(self) -> None:
        process = mock.Mock()
        process.poll.return_value = None
        evidence_error = performance.pacing.WslPacingEvidenceError(
            "synthetic malformed stream"
        )
        with mock.patch.object(
            performance.subprocess,
            "Popen",
            return_value=process,
        ), mock.patch.object(
            performance,
            "_build_elapsed_seconds",
            side_effect=evidence_error,
        ), mock.patch.object(
            performance,
            "_terminate_build_process",
        ) as terminate:
            with self.assertRaisesRegex(
                performance.pacing.WslPacingEvidenceError,
                "synthetic malformed stream",
            ):
                performance.build_binary(performance.time.monotonic() + 60.0)
        terminate.assert_called_once_with(process)

    def test_build_elapsed_subtracts_only_verified_overlap(self) -> None:
        snapshot = mock.Mock()
        snapshot.overlap_seconds.return_value = 500.0
        with mock.patch.object(
            performance.pacing,
            "read_pacing_snapshot",
            return_value=snapshot,
        ) as read:
            observed = performance._build_elapsed_seconds(
                100.0,
                700.0,
                {"fixture": "environment"},
            )
        self.assertEqual(observed, (600.0, 500.0, 100.0))
        read.assert_called_once_with(
            {"fixture": "environment"},
            expected_policy_sha256=performance.THERMAL_POLICY_CONTENT_SHA256,
        )

    def test_build_termination_escalates_to_process_group_kill(self) -> None:
        process = mock.Mock()
        process.pid = 1234
        process.poll.return_value = None
        process.wait.side_effect = [
            performance.subprocess.TimeoutExpired(["cargo"], 15.0),
            0,
        ]
        with mock.patch.object(performance.os, "killpg") as killpg:
            performance._terminate_build_process(process)
        self.assertEqual(
            killpg.call_args_list,
            [
                mock.call(1234, performance.signal.SIGTERM),
                mock.call(1234, performance.signal.SIGKILL),
            ],
        )

    def test_summary_requires_hash_bound_passing_receipts(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            rows = []
            receipts: dict[Path, dict] = {}
            for profile in performance.PROFILES:
                path = root / f"{profile}.kiln.json"
                path.write_text("{}")
                rows.append(
                    {
                        "profile": profile,
                        "status": "completed",
                        "exit_code": 0,
                        "receipt": str(path),
                        "receipt_sha256": performance.sha256_file(path),
                        "blocked_by_profile": None,
                    }
                )
                receipts[path] = {
                    "engine": {"name": "kiln"},
                    "verdict": "passed",
                    "workload": {
                        "profile": profile,
                        "concurrency": [1],
                        "repeats": 1,
                        "max_tokens": 64,
                    },
                    "runs": [
                        {
                            "completion_tokens": 64,
                            "elapsed_s": 2.0,
                            "success_count": 1,
                            "error_count": 0,
                            "memory": {"peak_bytes": 100},
                        }
                    ],
                }
            summary = {
                "schema": "kiln.serving-benchmark-campaign.v9",
                "created_at": "2026-07-25T00:00:00+00:00",
                "campaign_id": "fixture-campaign",
                "prompt_set_id": performance.PROMPT_SET_ID,
                "engine": "kiln",
                "reference_role": "qualification_gate",
                "reference_dir": None,
                "output_evidence": "hashes",
                "model_fingerprint_read_mib_per_second": 64,
                "execution_policy": "continue_after_failure",
                "memory_sampler": {},
                "thermal_policy": {},
                "server_owner": {},
                "profiles": rows,
                "verdict": "passed",
            }
            summary["summary_sha256"] = performance.canonical_sha256(summary)
            (root / "campaign.kiln.json").write_text(json.dumps(summary))
            metrics, hashes = performance.summarize_campaign(
                root,
                "kiln",
                receipt_loader=lambda path: receipts[path],
            )
            self.assertEqual(metrics["profile_pass_count"], 5)
            self.assertEqual(metrics["completion_token_count"], 320)
            self.assertEqual(metrics["output_token_throughput_per_second"], 32.0)
            self.assertEqual(hashes, [performance.sha256_file(root / f"{p}.kiln.json") for p in performance.PROFILES])


if __name__ == "__main__":
    unittest.main()
