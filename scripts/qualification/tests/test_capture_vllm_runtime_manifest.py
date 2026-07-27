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


def load_wsl_supervision(policy_name: str) -> object:
    def repository_file(path: Path, _label: str) -> tuple[Path, str]:
        absolute = path if path.is_absolute() else ROOT / path
        absolute = absolute.absolute()
        return absolute, absolute.relative_to(ROOT).as_posix()

    with mock.patch.object(
        capture,
        "_repository_regular_file",
        side_effect=repository_file,
    ):
        thermal = capture.load_wsl2_thermal_supervision(
            Path("qualification/host-policies") / policy_name
        )
    return capture.Wsl2ScopeSupervision(
        unshare_path=thermal.unshare_path,
        thermal=thermal,
    )


def write_capture_program(path: Path, body: str) -> None:
    path.write_text("#!/bin/sh\nset -eu\n" + body, encoding="utf-8")
    path.chmod(0o755)


def thermal_stderr(supervision: object) -> bytes:
    policy = supervision.thermal.policy
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


def scope_stderr(supervision: object) -> bytes:
    unit = "kiln-wsl-scope-" + "a" * 32
    duration_seconds = 10.0
    cpu_allowed_usec = 5_000_000
    policy = supervision.thermal.policy
    pacing = policy.pacing
    assert pacing is not None
    start = {
        "schema": capture.WSL_SCOPE_EVENT_SCHEMA,
        "event": "start",
        "unit": unit,
        "cgroup": (
            f"/sys/fs/cgroup/user.slice/user-{capture.os.getuid()}.slice/"
            f"user@{capture.os.getuid()}.service/app.slice/{unit}.scope"
        ),
        "containment": capture.benchmark.WSL2_NETWORK_BOUNDARY,
        "memory_max_bytes": capture.benchmark.WSL2_SCOPE_MEMORY_MAX_BYTES,
        "memory_swap_max_bytes": 0,
        "pids_max": capture.benchmark.WSL2_SCOPE_PIDS_MAX,
        "cpu_quota_percent": capture.benchmark.WSL2_SCOPE_CPU_QUOTA_PERCENT,
        "cpu_controller": "usage-feedback-cgroup-freeze-v1",
        "cpu_poll_interval_ms": capture.WSL_SCOPE_CPU_POLL_INTERVAL_MS,
        "runtime_max_seconds": 1470.0,
        "thermal_policy_sha256": policy.content_sha256,
        "thermal_pacing": {
            "policy_sha256": policy.content_sha256,
            "mode": pacing.mode,
            "telemetry_source": "outer-supervisor-inherited-pipe-v1",
            "freeze_verification": "cgroup-freeze-and-events-roundtrip-v1",
            "host_start_millicelsius": pacing.host_start_millicelsius,
            "host_resume_millicelsius": pacing.host_resume_millicelsius,
            "gpu_start_millicelsius": pacing.gpu_start_millicelsius,
            "gpu_resume_millicelsius": pacing.gpu_resume_millicelsius,
            "resume_stable_samples": pacing.resume_stable_samples,
            "timeout_seconds": pacing.timeout_seconds,
        },
    }
    complete = {
        "schema": capture.WSL_SCOPE_EVENT_SCHEMA,
        "event": "complete",
        "unit": unit,
        "duration_seconds": duration_seconds,
        "cpu_usage_usec": cpu_allowed_usec - 1,
        "cpu_allowed_usec": cpu_allowed_usec,
        "cpu_quota_percent": capture.benchmark.WSL2_SCOPE_CPU_QUOTA_PERCENT,
        "memory_peak_bytes": 1024 * 1024,
        "memory_events": {
            "low": 0,
            "high": 0,
            "max": 0,
            "oom": 0,
            "oom_kill": 0,
            "oom_group_kill": 0,
        },
        "pids_peak": 2,
        "scope_removed": True,
        "child_returncode": 0,
        "reason": None,
        "thermal_pacing": {
            "policy_sha256": policy.content_sha256,
            "mode": pacing.mode,
            "active": False,
            "sample_count": 10,
            "pause_count": 1,
            "completed_pause_count": 1,
            "total_pause_seconds": 3.0,
            "longest_pause_seconds": 3.0,
            "peak_host_millicelsius": 80_000,
            "peak_gpu_millicelsius": 65_000,
            "ending_host_millicelsius": 72_000,
            "ending_gpu_millicelsius": 64_000,
        },
    }
    return (
        f"{capture.WSL_SCOPE_EVENT_PREFIX}"
        f"{json.dumps(start, separators=(',', ':'))}\n"
        f"{capture.WSL_SCOPE_EVENT_PREFIX}"
        f"{json.dumps(complete, separators=(',', ':'))}\n"
    ).encode()


def unpaced_scope_stderr() -> bytes:
    unit = "kiln-wsl-scope-" + "b" * 32
    start = {
        "schema": capture.WSL_SCOPE_EVENT_SCHEMA_UNPACED,
        "event": "start",
        "unit": unit,
        "cgroup": (
            f"/sys/fs/cgroup/user.slice/user-{capture.os.getuid()}.slice/"
            f"user@{capture.os.getuid()}.service/app.slice/{unit}.scope"
        ),
        "containment": capture.benchmark.WSL2_NETWORK_BOUNDARY,
        "memory_max_bytes": capture.WSL_SCOPE_UNPACED_MEMORY_MAX_BYTES,
        "memory_swap_max_bytes": 0,
        "pids_max": capture.benchmark.WSL2_SCOPE_PIDS_MAX,
        "cpu_quota_percent": 0,
        "cpu_controller": "not_configured",
        "cpu_poll_interval_ms": capture.WSL_SCOPE_CPU_POLL_INTERVAL_MS,
        "runtime_max_seconds": 1770.0,
        "thermal_policy_sha256": None,
    }
    complete = {
        "schema": capture.WSL_SCOPE_EVENT_SCHEMA_UNPACED,
        "event": "complete",
        "unit": unit,
        "duration_seconds": 10.0,
        "cpu_usage_usec": 5_000_001,
        "cpu_allowed_usec": None,
        "cpu_quota_percent": 0,
        "memory_peak_bytes": 1024 * 1024,
        "memory_events": {
            "low": 0,
            "high": 0,
            "max": 0,
            "oom": 0,
            "oom_kill": 0,
            "oom_group_kill": 0,
        },
        "pids_peak": 2,
        "scope_removed": True,
        "child_returncode": 0,
        "reason": None,
    }
    return (
        f"{capture.WSL_SCOPE_EVENT_PREFIX}"
        f"{json.dumps(start, separators=(',', ':'))}\n"
        f"{capture.WSL_SCOPE_EVENT_PREFIX}"
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
        supervision = load_wsl_supervision(
            "rtx4090-laptop-wsl2-cgroup-pacing-v2.json"
        )
        thermal = supervision.thermal
        self.assertIsNotNone(thermal)
        config = launch_config(Path("/tmp/fixture-python"))
        command = capture.supervised_manifest_command(config, supervision, 1800.0)
        self.assertEqual(
            command[:5],
            [
                sys.executable,
                str(capture.WSL_THERMAL_EXEC),
                "--policy",
                str(thermal.path),
                "--",
            ],
        )
        self.assertEqual(command[5:7], [sys.executable, str(capture.WSL_SCOPE_EXEC)])
        self.assertIn(str(capture.benchmark.WSL2_SCOPE_MEMORY_MAX_BYTES), command)
        self.assertIn(str(capture.benchmark.WSL2_SCOPE_CPU_QUOTA_PERCENT), command)
        pacing_index = command.index("--thermal-pacing-policy")
        self.assertEqual(command[pacing_index + 1], str(thermal.path))
        self.assertIn(str(supervision.unshare_path), command)
        self.assertIn(str(capture.LINUX_NAMESPACE_EXEC), command)
        self.assertIn("/tmp/fixture-python", command)
        self.assertIn("--manifest-only", command)

        evidence = capture.validate_wsl2_thermal_stderr(
            thermal_stderr(supervision),
            thermal,
        )
        self.assertEqual(evidence["policy_sha256"], thermal.policy.content_sha256)
        self.assertEqual(evidence["ending_host_millicelsius"], 80_000)
        scope_evidence = capture.validate_wsl2_scope_stderr(
            scope_stderr(supervision),
            supervision,
            1470.0,
        )
        self.assertEqual(
            scope_evidence["memory_max_bytes"],
            capture.benchmark.WSL2_SCOPE_MEMORY_MAX_BYTES,
        )
        self.assertEqual(scope_evidence["cpu_usage_usec"], 4_999_999)
        self.assertTrue(scope_evidence["scope_removed"])
        self.assertEqual(scope_evidence["thermal_pacing"]["pause_count"], 1)

        mutated = thermal_stderr(supervision).replace(
            b'"supervision_outcome":"child_exit"',
            b'"supervision_outcome":"thermal_trip"',
        )
        with self.assertRaisesRegex(capture.CaptureError, "child_exit"):
            capture.validate_wsl2_thermal_stderr(mutated, thermal)
        scope_mutated = scope_stderr(supervision).replace(
            b'"scope_removed":true',
            b'"scope_removed":false',
        )
        with self.assertRaisesRegex(capture.CaptureError, "required lifecycle"):
            capture.validate_wsl2_scope_stderr(
                scope_mutated,
                supervision,
                1470.0,
            )
        pacing_mutated = scope_stderr(supervision).replace(
            b'"active":false',
            b'"active":true',
        )
        with self.assertRaisesRegex(capture.CaptureError, "finish inactive"):
            capture.validate_wsl2_scope_stderr(
                pacing_mutated,
                supervision,
                1470.0,
            )
        ordered = (
            thermal_stderr(supervision).splitlines()[1]
            + b"\n"
            + scope_stderr(supervision)
            + thermal_stderr(supervision).splitlines()[2]
            + b"\n"
        )
        capture.validate_wsl2_event_order(ordered, thermal_requested=True)
        with self.assertRaisesRegex(capture.CaptureError, "event order"):
            capture.validate_wsl2_event_order(
                scope_stderr(supervision) + thermal_stderr(supervision),
                thermal_requested=True,
            )

    def test_wsl2_unpaced_supervision_keeps_scope_without_thermal_or_cpu_quota(
        self,
    ) -> None:
        supervision = capture.Wsl2ScopeSupervision(
            unshare_path=Path("/usr/bin/unshare"),
            thermal=None,
        )
        command = capture.supervised_manifest_command(
            launch_config(Path("/tmp/fixture-python")),
            supervision,
            1800.0,
        )
        self.assertEqual(command[:2], [sys.executable, str(capture.WSL_SCOPE_EXEC)])
        quota_index = command.index("--cpu-quota-percent")
        self.assertEqual(command[quota_index + 1], "0")
        memory_index = command.index("--memory-max-bytes")
        self.assertEqual(
            command[memory_index + 1],
            str(capture.WSL_SCOPE_UNPACED_MEMORY_MAX_BYTES),
        )
        self.assertNotIn("--thermal-pacing-policy", command)
        self.assertNotIn(str(capture.WSL_THERMAL_EXEC), command)
        self.assertIn(str(supervision.unshare_path), command)

        evidence = capture.validate_wsl2_scope_stderr(
            unpaced_scope_stderr(),
            supervision,
            1770.0,
        )
        self.assertEqual(evidence["mechanism"], "systemd-user-scope-v1")
        self.assertEqual(evidence["memory_max_bytes"], 0)
        self.assertEqual(evidence["cpu_quota_percent"], 0)
        self.assertIsNone(evidence["cpu_allowed_usec"])
        self.assertIsNone(evidence["thermal_pacing"])
        capture.validate_wsl2_event_order(
            unpaced_scope_stderr(),
            thermal_requested=False,
        )

    def test_wsl2_policy_must_be_a_repository_regular_file(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            outside = Path(directory) / "policy.json"
            outside.write_text("{}")
            with self.assertRaisesRegex(capture.CaptureError, "inside"):
                capture.load_wsl2_thermal_supervision(outside)

    def test_wsl2_capture_rejects_hard_limit_only_v1_policy(self) -> None:
        with self.assertRaisesRegex(capture.CaptureError, "requires a v2"):
            load_wsl_supervision(
                "rtx4090-laptop-wsl2-boundary-v1.json"
            )

    def test_wsl2_platform_uses_unpaced_scope_without_thermal_policy(self) -> None:
        with mock.patch.object(
            capture.platform,
            "release",
            return_value="6.6.87.2-microsoft-standard-WSL2",
        ), mock.patch.object(
            capture,
            "_load_wsl2_scope_prerequisites",
            return_value=Path("/usr/bin/unshare"),
        ):
            supervision = capture.load_platform_supervision(None)
            self.assertEqual(supervision.unshare_path, Path("/usr/bin/unshare"))
            self.assertIsNone(supervision.thermal)
        with mock.patch.object(
            capture.platform,
            "release",
            return_value="6.8.0-generic",
        ):
            with self.assertRaisesRegex(capture.CaptureError, "only valid on WSL2"):
                capture.load_platform_supervision(Path("policy.json"))
            self.assertIsNone(capture.load_platform_supervision(None))

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
