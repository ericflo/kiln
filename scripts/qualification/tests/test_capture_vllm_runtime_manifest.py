from __future__ import annotations

import base64
import hashlib
import importlib.util
import json
import shlex
import sys
import tempfile
import unittest
from pathlib import Path


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
