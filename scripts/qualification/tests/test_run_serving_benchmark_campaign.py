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

    def test_command_pairs_profile_with_matching_kiln_receipt(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            args = campaign.parse_args(required_args(root, "vllm"))
            command = campaign.benchmark_command(
                args, "mixed", root / "mixed.vllm.json"
            )
        self.assertEqual(command[command.index("--workload-profile") + 1], "mixed")
        self.assertEqual(
            command[command.index("--reference-receipt") + 1],
            str(root / "kiln" / "mixed.kiln.json"),
        )
        self.assertEqual(
            command[command.index("--host-thermal-policy") + 1],
            str(root / "host-thermal-policy.json"),
        )
        self.assertEqual(command[command.index("--server-pid") + 1], "4321")

    def test_campaign_runs_every_profile_and_self_hashes_summary(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)

            def fake_run(command: list[str], check: bool) -> SimpleNamespace:
                self.assertFalse(check)
                output = Path(command[command.index("--out") + 1])
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
            self.assertEqual(summary["server_owner"]["server_pid"], 4321)
            self.assertEqual(
                summary["server_owner"]["mode"], "attached_process_group"
            )
            self.assertTrue(
                summary["host_thermal_policy"]["sha256"].startswith("sha256:")
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


if __name__ == "__main__":
    unittest.main()
