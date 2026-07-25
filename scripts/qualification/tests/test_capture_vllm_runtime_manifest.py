from __future__ import annotations

import base64
import hashlib
import importlib.util
import json
import shlex
import signal
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock


ROOT = Path(__file__).resolve().parents[3]
SCRIPT = ROOT / "scripts" / "qualification" / "capture_vllm_runtime_manifest.py"
SPEC = importlib.util.spec_from_file_location("capture_vllm_runtime_manifest", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
capture = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = capture
SPEC.loader.exec_module(capture)


def manifest(runtime_hash: str = "a" * 64) -> bytes:
    identity = {
        "schema": "kiln.teacher-identity.v1",
        "served_model_id": "Qwen3.5-4B",
        "implementation": "vllm:0.25.0",
        "max_top_k": 20,
        "max_model_len": 32_768,
    }
    canonical = json.dumps(
        identity,
        ensure_ascii=False,
        allow_nan=False,
        separators=(",", ":"),
    )
    encoded = base64.urlsafe_b64encode(canonical.encode()).decode().rstrip("=")
    value = {
        "identity": identity,
        "canonical_json": canonical,
        "system_fingerprint": (
            f"kiln-teacher-v1.{encoded}."
            f"{hashlib.sha256(canonical.encode()).hexdigest()}"
        ),
        "runtime_content_sha256": runtime_hash,
    }
    return (json.dumps(value, indent=2) + "\n").encode()


def launch_config(executable: Path) -> object:
    return capture.benchmark.ServerLaunchConfig(
        record={"id": "fixture-vllm-launch"},
        command=(
            str(executable),
            "scripts/vllm_teacher.py",
            "--model-path=Qwen3.5-4B",
            "--served-model-id=Qwen3.5-4B",
            "--process-group-mode=inherited",
            "--snapshot-root=.qualification/snapshots",
            "--cache-root=.qualification/caches",
            "--max-top-k=20",
            "--max-model-len=32768",
            "--",
            "--host=127.0.0.1",
            "--port=8421",
            "--dtype=bfloat16",
        ),
        working_directory=ROOT,
        log_directory=ROOT / ".qualification" / "logs",
        readiness_poll_interval_seconds=0.25,
        startup_timeout_seconds=1200.0,
        shutdown_timeout_seconds=180.0,
        acceptable_exit_codes=(0,),
    )


def write_capture_program(path: Path, body: str) -> None:
    path.write_text("#!/bin/sh\nset -eu\n" + body, encoding="utf-8")
    path.chmod(0o755)


def thermal_stderr(supervision: object) -> bytes:
    policy = supervision.policy
    preflight = {
        "schema": capture.WSL_THERMAL_EVENT_SCHEMA,
        "event": "preflight",
        "policy_id": policy.policy_id,
        "policy_sha256": policy.content_sha256,
        "host_millicelsius": 70_000,
        "gpu_millicelsius": 60_000,
        "host_limit_millicelsius": policy.host_limit_millicelsius,
        "gpu_limit_millicelsius": policy.gpu_limit_millicelsius,
    }
    complete = {
        "schema": capture.WSL_THERMAL_EVENT_SCHEMA,
        "event": "complete",
        "policy_id": policy.policy_id,
        "policy_sha256": policy.content_sha256,
        "supervision_outcome": "child_exit",
        "failure_reason": None,
        "child_returncode": 0,
        "sample_count": 10,
        "starting_host_millicelsius": 70_000,
        "starting_gpu_millicelsius": 60_000,
        "peak_host_millicelsius": 90_000,
        "peak_gpu_millicelsius": 65_000,
        "ending_host_millicelsius": 80_000,
        "ending_gpu_millicelsius": 64_000,
        "safe_handoff_stable_samples": policy.handoff_stable_samples,
    }
    return (
        "runtime warning\n"
        f"{capture.WSL_THERMAL_EVENT_PREFIX}"
        f"{json.dumps(preflight, separators=(',', ':'))}\n"
        f"{capture.WSL_THERMAL_EVENT_PREFIX}"
        f"{json.dumps(complete, separators=(',', ':'))}\n"
    ).encode()


class VllmRuntimeManifestCaptureTests(unittest.TestCase):
    def test_manifest_command_inserts_only_manifest_mode(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            executable = Path(directory) / "capture"
            write_capture_program(executable, "exit 0\n")
            config = launch_config(executable)
            command = capture.manifest_command(config)
        boundary = command.index("--")
        self.assertEqual(command[boundary - 1], "--manifest-only")
        self.assertNotIn("--manifest-only", config.command)
        self.assertEqual(command[: boundary - 1], list(config.command[:9]))
        self.assertEqual(command[boundary + 1 :], list(config.command[10:]))

    def test_wsl2_supervision_wraps_each_manifest_command_and_validates_handoff(
        self,
    ) -> None:
        supervision = capture.load_wsl2_thermal_supervision(
            Path(
                "qualification/host-policies/"
                "rtx4090-laptop-wsl2-boundary-v1.json"
            )
        )
        config = launch_config(Path("/tmp/fixture-python"))
        command = capture.supervised_manifest_command(config, supervision)
        self.assertEqual(
            command[:6],
            [
                sys.executable,
                str(capture.WSL_THERMAL_EXEC),
                "--policy",
                str(supervision.path),
                "--",
                "/tmp/fixture-python",
            ],
        )
        self.assertIn("--manifest-only", command)

        evidence = capture.validate_wsl2_thermal_stderr(
            thermal_stderr(supervision),
            supervision,
        )
        self.assertEqual(evidence["policy_sha256"], supervision.policy.content_sha256)
        self.assertEqual(evidence["ending_host_millicelsius"], 80_000)

        mutated = thermal_stderr(supervision).replace(
            b'"supervision_outcome":"child_exit"',
            b'"supervision_outcome":"thermal_trip"',
        )
        with self.assertRaisesRegex(capture.CaptureError, "child_exit"):
            capture.validate_wsl2_thermal_stderr(mutated, supervision)

    def test_wsl2_policy_must_be_a_repository_regular_file(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            outside = Path(directory) / "policy.json"
            outside.write_text("{}")
            with self.assertRaisesRegex(capture.CaptureError, "inside"):
                capture.load_wsl2_thermal_supervision(outside)

    def test_wsl2_platform_requires_thermal_supervision(self) -> None:
        with mock.patch.object(
            capture.platform,
            "release",
            return_value="6.6.87.2-microsoft-standard-WSL2",
        ):
            with self.assertRaisesRegex(capture.CaptureError, "required on WSL2"):
                capture.load_platform_thermal_supervision(None)
        with mock.patch.object(
            capture.platform,
            "release",
            return_value="6.8.0-generic",
        ):
            with self.assertRaisesRegex(capture.CaptureError, "only valid on WSL2"):
                capture.load_platform_thermal_supervision(Path("policy.json"))
            self.assertIsNone(capture.load_platform_thermal_supervision(None))

    def test_capture_timeout_terminates_the_complete_child_session(self) -> None:
        process = mock.Mock()
        process.pid = 4242
        process.poll.return_value = None
        process.wait.side_effect = [
            subprocess.TimeoutExpired(["fixture"], 5.0),
            143,
        ]
        with mock.patch.object(
            capture.subprocess,
            "Popen",
            return_value=process,
        ) as popen, mock.patch.object(capture.os, "killpg") as killpg:
            with self.assertRaisesRegex(capture.CaptureError, "exceeded"):
                capture._run_capture_child(
                    ["fixture"],
                    working_directory=ROOT,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    timeout_seconds=5.0,
                    termination_grace_seconds=30.0,
                )
        self.assertTrue(popen.call_args.kwargs["start_new_session"])
        killpg.assert_called_once_with(4242, signal.SIGTERM)
        self.assertEqual(process.wait.call_args_list[-1], mock.call(timeout=30.0))

    def test_two_identical_captures_publish_exact_bytes_without_clobber(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            payload_path = root / "manifest.json"
            payload_path.write_bytes(manifest())
            executable = root / "capture"
            write_capture_program(
                executable,
                f"cat {shlex.quote(str(payload_path))}\n",
            )
            first, second = capture.capture_twice(
                launch_config(executable),
                timeout_seconds=5.0,
            )
            output = root / "published.json"
            capture.publish_no_clobber(output, first.payload)
            self.assertEqual(first.payload, second.payload)
            self.assertEqual(output.read_bytes(), manifest())
            self.assertEqual(output.stat().st_mode & 0o777, 0o644)
            with self.assertRaisesRegex(capture.CaptureError, "refusing to replace"):
                capture.publish_no_clobber(output, first.payload)

    def test_capture_rejects_nonrepeatable_runtime_identity(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            first_path = root / "first.json"
            second_path = root / "second.json"
            marker = root / "marker"
            first_path.write_bytes(manifest("a" * 64))
            second_path.write_bytes(manifest("b" * 64))
            executable = root / "capture"
            write_capture_program(
                executable,
                "if [ -e {marker} ]; then cat {second}; "
                "else : > {marker}; cat {first}; fi\n".format(
                    marker=shlex.quote(str(marker)),
                    first=shlex.quote(str(first_path)),
                    second=shlex.quote(str(second_path)),
                ),
            )
            with self.assertRaisesRegex(capture.CaptureError, "not byte-identical"):
                capture.capture_twice(
                    launch_config(executable),
                    timeout_seconds=5.0,
                )


if __name__ == "__main__":
    unittest.main()
