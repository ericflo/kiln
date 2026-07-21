from __future__ import annotations

import importlib.util
import io
import json
import sys
import tempfile
import unittest
from contextlib import redirect_stderr
from pathlib import Path
from types import SimpleNamespace
from unittest import mock


ROOT = Path(__file__).resolve().parents[3]
SCRIPT = ROOT / "scripts" / "run-serving-benchmark-campaign.py"
SPEC = importlib.util.spec_from_file_location("run_serving_benchmark_campaign", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
campaign = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = campaign
SPEC.loader.exec_module(campaign)


def required_args(directory: Path, engine: str = "kiln") -> list[str]:
    thermal_policy = directory / "host-thermal-policy.json"
    thermal_policy.write_text('{"fixture":true}\n')
    args = [
        "--engine",
        engine,
        "--base-url",
        "http://127.0.0.1:8420",
        "--model-path",
        str(directory / "model"),
        "--runtime-identity",
        "fixture-runtime",
        "--runtime-artifact",
        str(directory / "runtime"),
        "--campaign-id",
        "fixture-v1",
        "--prompt-set-id",
        "shared-prompts-v1",
        "--out-dir",
        str(directory / "out"),
        "--memory-path",
        str(directory / "memory"),
        "--memory-limit-bytes",
        "4096",
        "--host-thermal-policy",
        str(thermal_policy),
        "--server-pid",
        "4321",
    ]
    if engine == "vllm":
        args.extend(("--reference-dir", str(directory / "kiln")))
    return args


class ServingBenchmarkCampaignTests(unittest.TestCase):
    def test_vllm_campaign_requires_reference_receipts(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            args = required_args(Path(directory), "vllm")
            reference_index = args.index("--reference-dir")
            del args[reference_index : reference_index + 2]
            with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                campaign.parse_args(args)

    def test_same_artifact_discriminator_requires_kiln_reference_receipts(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                campaign.parse_args(
                    [
                        *required_args(root),
                        "--reference-role",
                        "same_artifact_graph_eager_discriminator",
                    ]
                )

            with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                campaign.parse_args(
                    [
                        *required_args(root, "vllm"),
                        "--reference-role",
                        "same_artifact_graph_eager_discriminator",
                    ]
                )

    def test_kiln_reference_dir_requires_discriminator_role(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                campaign.parse_args(
                    [
                        *required_args(root),
                        "--reference-dir",
                        str(root / "eager"),
                    ]
                )

    def test_campaign_bounds_model_fingerprint_read_rate(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            args = required_args(Path(directory))
            with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                campaign.parse_args(
                    [*args, "--model-fingerprint-read-mib-per-second", "63"]
                )

    def test_command_pairs_profile_with_matching_kiln_receipt(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            args = campaign.parse_args(required_args(root, "vllm"))
            command = campaign.benchmark_command(
                args, "mixed", root / "mixed.vllm.json"
            )
        self.assertEqual(command[command.index("--workload-profile") + 1], "mixed")
        self.assertEqual(
            command[command.index("--run-id") + 1],
            "fixture-v1-vllm-mixed",
        )
        self.assertEqual(
            command[command.index("--prompt-set-id") + 1],
            "shared-prompts-v1-mixed",
        )
        self.assertEqual(
            command[command.index("--reference-receipt") + 1],
            str(root / "kiln" / "mixed.kiln.json"),
        )
        self.assertEqual(
            command[command.index("--host-thermal-policy") + 1],
            str(root / "host-thermal-policy.json"),
        )
        self.assertEqual(command[command.index("--server-pid") + 1], "4321")
        self.assertEqual(command[command.index("--memory-source") + 1], "drm")
        self.assertEqual(
            command[command.index("--memory-path") + 1], str(root / "memory")
        )
        self.assertEqual(
            command[command.index("--output-evidence") + 1], "hashes"
        )
        self.assertEqual(
            command[command.index("--reference-role") + 1],
            "qualification_gate",
        )
        self.assertEqual(
            command[
                command.index("--model-fingerprint-read-mib-per-second") + 1
            ],
            "256",
        )

    def test_nvml_device_selection_is_typed_and_forwarded(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            arguments = required_args(root)
            memory_path = arguments.index("--memory-path")
            del arguments[memory_path : memory_path + 2]
            arguments.extend(("--memory-source", "nvml", "--memory-device-index", "1"))
            args = campaign.parse_args(arguments)
            command = campaign.benchmark_command(
                args, "mixed", root / "mixed.kiln.json"
            )
        self.assertEqual(args.memory_source, "nvml")
        self.assertEqual(command[command.index("--memory-device-index") + 1], "1")
        self.assertEqual(command[command.index("--memory-path") + 1], "auto")

    def test_nvml_uuid_selection_is_forwarded_without_an_index(self) -> None:
        uuid = "GPU-01234567-89ab-cdef-0123-456789abcdef"
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            arguments = required_args(root)
            memory_path = arguments.index("--memory-path")
            del arguments[memory_path : memory_path + 2]
            arguments.extend(("--memory-device-uuid", uuid))
            args = campaign.parse_args(arguments)
            command = campaign.benchmark_command(
                args, "mixed", root / "mixed.kiln.json"
            )
        self.assertEqual(args.memory_source, "nvml")
        self.assertEqual(command[command.index("--memory-device-uuid") + 1], uuid)
        self.assertNotIn("--memory-device-index", command)

    def test_campaign_rejects_conflicting_memory_selectors(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            arguments = required_args(Path(directory))
            arguments.extend(("--memory-device-index", "0"))
            with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                campaign.parse_args(arguments)

    def test_command_pairs_graph_campaign_with_same_artifact_eager_receipt(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            args = campaign.parse_args(
                [
                    *required_args(root),
                    "--reference-role",
                    "same_artifact_graph_eager_discriminator",
                    "--reference-dir",
                    str(root / "eager"),
                ]
            )
            command = campaign.benchmark_command(
                args, "greedy-short", root / "greedy-short.kiln.json"
            )

        self.assertEqual(
            command[command.index("--reference-receipt") + 1],
            str(root / "eager" / "greedy-short.kiln.json"),
        )
        self.assertEqual(
            command[command.index("--reference-role") + 1],
            "same_artifact_graph_eager_discriminator",
        )

    def test_campaigns_keep_prompt_identity_stable_and_run_identity_unique(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            kiln_args = campaign.parse_args(required_args(root, "kiln"))
            vllm_args = campaign.parse_args(required_args(root, "vllm"))
            kiln_command = campaign.benchmark_command(
                kiln_args, "greedy-short", root / "kiln.json"
            )
            vllm_command = campaign.benchmark_command(
                vllm_args, "greedy-short", root / "vllm.json"
            )

        self.assertNotEqual(
            kiln_command[kiln_command.index("--run-id") + 1],
            vllm_command[vllm_command.index("--run-id") + 1],
        )
        self.assertEqual(
            kiln_command[kiln_command.index("--prompt-set-id") + 1],
            vllm_command[vllm_command.index("--prompt-set-id") + 1],
        )

    def test_campaign_runs_every_profile_and_self_hashes_summary(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            invoked_outputs: list[Path] = []

            def fake_run(command: list[str], check: bool) -> SimpleNamespace:
                self.assertFalse(check)
                output = Path(command[command.index("--out") + 1])
                invoked_outputs.append(output)
                self.assertNotEqual(output.parent, root / "out")
                self.assertEqual(list((root / "out").glob("*.json")), [])
                output.write_text('{"fixture":true}\n')
                return SimpleNamespace(returncode=0)

            with mock.patch.object(campaign.subprocess, "run", side_effect=fake_run):
                self.assertEqual(campaign.main(required_args(root)), 0)

            summary_path = root / "out" / "campaign.kiln.json"
            summary = json.loads(summary_path.read_text())
            recorded_hash = summary.pop("summary_sha256")
            self.assertEqual(recorded_hash, campaign.canonical_sha256(summary))
            self.assertEqual(
                [row["profile"] for row in summary["profiles"]],
                list(campaign.PROFILES),
            )
            self.assertEqual(summary["verdict"], "passed")
            self.assertEqual(summary["schema"], campaign.SCHEMA)
            self.assertEqual(summary["prompt_set_id"], "shared-prompts-v1")
            self.assertEqual(summary["reference_role"], "qualification_gate")
            self.assertIsNone(summary["reference_dir"])
            self.assertEqual(summary["execution_policy"], "fail_fast")
            self.assertEqual(
                summary["model_fingerprint_read_mib_per_second"], 256
            )
            self.assertEqual(summary["server_owner"]["server_pid"], 4321)
            self.assertEqual(summary["output_evidence"], "hashes")
            self.assertEqual(
                summary["memory_sampler"],
                {
                    "source": "drm",
                    "path": str((root / "memory").resolve()),
                    "device_index": None,
                    "device_uuid": None,
                    "interval_ms": 50,
                    "limit_bytes": 4096,
                },
            )
            self.assertEqual(
                summary["server_owner"]["mode"], "attached_process_group"
            )
            self.assertTrue(
                summary["host_thermal_policy"]["sha256"].startswith("sha256:")
            )
            self.assertEqual(len(invoked_outputs), len(campaign.PROFILES))
            for row in summary["profiles"]:
                self.assertEqual(row["status"], "completed")
                self.assertIsNone(row["blocked_by_profile"])
                self.assertTrue(Path(row["receipt"]).is_file())

    def test_campaign_stops_after_first_failed_profile_and_publishes_evidence(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            calls: list[str] = []

            def fake_run(command: list[str], check: bool) -> SimpleNamespace:
                self.assertFalse(check)
                profile = command[command.index("--workload-profile") + 1]
                calls.append(profile)
                output = Path(command[command.index("--out") + 1])
                output.write_text('{"failed_counterexample":true}\n')
                return SimpleNamespace(returncode=2)

            with mock.patch.object(campaign.subprocess, "run", side_effect=fake_run):
                self.assertEqual(campaign.main(required_args(root)), 2)

            self.assertEqual(calls, [campaign.PROFILES[0]])
            summary = json.loads(
                (root / "out" / "campaign.kiln.json").read_text()
            )
            first, *skipped = summary["profiles"]
            self.assertEqual(first["status"], "completed")
            self.assertEqual(first["exit_code"], 2)
            self.assertTrue(first["receipt_sha256"].startswith("sha256:"))
            self.assertTrue(Path(first["receipt"]).is_file())
            self.assertEqual(summary["verdict"], "failed")
            for row in skipped:
                self.assertEqual(row["status"], "not_run_after_failure")
                self.assertIsNone(row["exit_code"])
                self.assertIsNone(row["receipt_sha256"])
                self.assertEqual(row["blocked_by_profile"], campaign.PROFILES[0])
                self.assertFalse(Path(row["receipt"]).exists())

    def test_campaign_can_explicitly_continue_after_failure(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            calls: list[str] = []

            def fake_run(command: list[str], check: bool) -> SimpleNamespace:
                self.assertFalse(check)
                profile = command[command.index("--workload-profile") + 1]
                calls.append(profile)
                output = Path(command[command.index("--out") + 1])
                output.write_text('{"fixture":true}\n')
                return SimpleNamespace(returncode=2 if len(calls) == 1 else 0)

            with mock.patch.object(campaign.subprocess, "run", side_effect=fake_run):
                self.assertEqual(
                    campaign.main(
                        [*required_args(root), "--continue-after-failure"]
                    ),
                    2,
                )

            self.assertEqual(calls, list(campaign.PROFILES))
            summary = json.loads(
                (root / "out" / "campaign.kiln.json").read_text()
            )
            self.assertEqual(summary["execution_policy"], "continue_after_failure")
            self.assertEqual(summary["verdict"], "failed")
            self.assertTrue(
                all(row["status"] == "completed" for row in summary["profiles"])
            )
            self.assertTrue(
                all(Path(row["receipt"]).is_file() for row in summary["profiles"])
            )

    def test_campaign_can_forward_owned_server_launch_config(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            raw_args = required_args(root)
            pid_index = raw_args.index("--server-pid")
            del raw_args[pid_index : pid_index + 2]
            launch_config = root / "server-launch.json"
            launch_config.write_text('{"fixture":true}\n')
            raw_args.extend(("--server-launch-config", str(launch_config)))
            args = campaign.parse_args(raw_args)
            command = campaign.benchmark_command(
                args, "greedy-short", root / "greedy-short.kiln.json"
            )

        self.assertNotIn("--server-pid", command)
        self.assertEqual(
            command[command.index("--server-launch-config") + 1],
            str(launch_config),
        )

    def test_campaign_can_request_full_output_evidence(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            args = campaign.parse_args(
                [*required_args(root), "--output-evidence", "full"]
            )
            command = campaign.benchmark_command(
                args, "greedy-short", root / "greedy-short.kiln.json"
            )
        self.assertEqual(
            command[command.index("--output-evidence") + 1], "full"
        )


if __name__ == "__main__":
    unittest.main()
