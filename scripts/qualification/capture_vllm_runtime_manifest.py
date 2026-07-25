#!/usr/bin/env python3
"""Capture a repeatable vLLM runtime manifest from an owned-launch document."""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import importlib.util
import json
import os
import platform
import resource
import signal
import stat
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Sequence


ROOT = Path(__file__).resolve().parents[2]
BENCHMARK_SCRIPT = ROOT / "scripts" / "bench-concurrent-batch.py"
BENCHMARK_SPEC = importlib.util.spec_from_file_location(
    "capture_vllm_benchmark_contract",
    BENCHMARK_SCRIPT,
)
assert BENCHMARK_SPEC is not None and BENCHMARK_SPEC.loader is not None
benchmark = importlib.util.module_from_spec(BENCHMARK_SPEC)
sys.modules[BENCHMARK_SPEC.name] = benchmark
BENCHMARK_SPEC.loader.exec_module(benchmark)


MAX_MANIFEST_BYTES = 1024 * 1024
MAX_STDERR_BYTES = 8 * 1024 * 1024
DEFAULT_TIMEOUT_SECONDS = 1800.0
CAPTURE_TERMINATION_GRACE_SECONDS = 30.0
WSL_THERMAL_EXEC = ROOT / "scripts" / "qualification" / "wsl_thermal_exec.py"
WSL_THERMAL_EVENT_PREFIX = "wsl2-thermal: "
WSL_THERMAL_EVENT_SCHEMA = "kiln.wsl2-thermal-event.v1"
WSL_THERMAL_PREFLIGHT_KEYS = {
    "schema",
    "event",
    "policy_id",
    "policy_sha256",
    "host_millicelsius",
    "gpu_millicelsius",
    "host_limit_millicelsius",
    "gpu_limit_millicelsius",
}
WSL_THERMAL_COMPLETE_KEYS = {
    "schema",
    "event",
    "policy_id",
    "policy_sha256",
    "supervision_outcome",
    "failure_reason",
    "child_returncode",
    "sample_count",
    "starting_host_millicelsius",
    "starting_gpu_millicelsius",
    "peak_host_millicelsius",
    "peak_gpu_millicelsius",
    "ending_host_millicelsius",
    "ending_gpu_millicelsius",
    "safe_handoff_stable_samples",
}


class CaptureError(RuntimeError):
    """Raised when a runtime manifest cannot be captured reproducibly."""


@dataclasses.dataclass(frozen=True)
class Capture:
    payload: bytes
    manifest: dict[str, Any]
    stderr_bytes: int
    stderr_sha256: str
    wsl2_thermal: dict[str, Any] | None


@dataclasses.dataclass(frozen=True)
class Wsl2ThermalSupervision:
    path: Path
    repository_path: str
    policy: Any


def require_clean_repository(root: Path = ROOT) -> str:
    """Require one clean committed source before runtime identity capture."""

    completed = subprocess.run(
        ["git", "status", "--porcelain=v1", "--untracked-files=normal"],
        cwd=root,
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    if completed.returncode != 0:
        details = completed.stderr.decode("utf-8", errors="replace").strip()
        raise CaptureError(f"cannot inspect repository state: {details}")
    if completed.stdout:
        raise CaptureError("repository must be clean before vLLM manifest capture")
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=root,
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    ).stdout.strip()
    if len(commit) != 40:
        raise CaptureError("repository HEAD is not a full commit hash")
    return commit


def manifest_command(config: Any) -> list[str]:
    """Insert manifest-only mode without duplicating the checked launch argv."""

    benchmark.validate_vllm_owned_launch(config)
    command = list(config.command)
    try:
        boundary = command.index("--", 2)
    except ValueError as exc:
        raise CaptureError("owned vLLM launch has no explicit argument boundary") from exc
    command.insert(boundary, "--manifest-only")
    return command


def _repository_regular_file(path: Path, label: str) -> tuple[Path, str]:
    absolute = Path(os.path.abspath(os.fspath(path if path.is_absolute() else ROOT / path)))
    try:
        repository_path = absolute.relative_to(ROOT).as_posix()
    except ValueError as exc:
        raise CaptureError(f"{label} must stay inside {ROOT}") from exc
    current = ROOT
    for part in Path(repository_path).parts:
        current = current / part
        try:
            info = current.lstat()
        except OSError as exc:
            raise CaptureError(f"cannot inspect {label} {current}: {exc}") from exc
        if stat.S_ISLNK(info.st_mode):
            raise CaptureError(f"{label} must not use symlinks: {current}")
    try:
        final_info = absolute.lstat()
    except OSError as exc:
        raise CaptureError(f"cannot inspect {label} {absolute}: {exc}") from exc
    if not stat.S_ISREG(final_info.st_mode):
        raise CaptureError(f"{label} must be a regular file: {absolute}")
    tracked = subprocess.run(
        ["git", "ls-files", "--error-unmatch", "--", repository_path],
        cwd=ROOT,
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    if tracked.returncode != 0:
        raise CaptureError(f"{label} must be tracked by the current repository")
    head_blob = subprocess.run(
        ["git", "rev-parse", f"HEAD:{repository_path}"],
        cwd=ROOT,
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    working_blob = subprocess.run(
        ["git", "hash-object", "--no-filters", "--", os.fspath(absolute)],
        cwd=ROOT,
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    if head_blob.returncode != 0 or working_blob.returncode != 0:
        raise CaptureError(f"cannot bind {label} to the current repository commit")
    if head_blob.stdout.strip() != working_blob.stdout.strip():
        raise CaptureError(f"{label} bytes do not match the current repository commit")
    return absolute, repository_path


def load_wsl2_thermal_supervision(path: Path) -> Wsl2ThermalSupervision:
    absolute, repository_path = _repository_regular_file(
        path,
        "WSL2 thermal policy",
    )
    _repository_regular_file(WSL_THERMAL_EXEC, "WSL2 thermal supervisor")
    try:
        policy = benchmark.wsl_thermal_exec.load_policy(absolute)
    except benchmark.wsl_thermal_exec.ThermalGuardError as exc:
        raise CaptureError(f"invalid WSL2 thermal policy: {exc}") from exc
    return Wsl2ThermalSupervision(
        path=absolute,
        repository_path=repository_path,
        policy=policy,
    )


def load_platform_thermal_supervision(
    path: Path | None,
) -> Wsl2ThermalSupervision | None:
    running_on_wsl2 = "microsoft-standard-wsl2" in platform.release().lower()
    if running_on_wsl2 and path is None:
        raise CaptureError("--wsl2-thermal-policy is required on WSL2")
    if not running_on_wsl2 and path is not None:
        raise CaptureError("--wsl2-thermal-policy is only valid on WSL2")
    return None if path is None else load_wsl2_thermal_supervision(path)


def supervised_manifest_command(
    config: Any,
    supervision: Wsl2ThermalSupervision | None,
) -> list[str]:
    command = manifest_command(config)
    if supervision is None:
        return command
    return [
        sys.executable,
        os.fspath(WSL_THERMAL_EXEC),
        "--policy",
        os.fspath(supervision.path),
        "--",
        *command,
    ]


def _limit_capture_files() -> None:
    resource.setrlimit(resource.RLIMIT_FSIZE, (MAX_STDERR_BYTES, MAX_STDERR_BYTES))


def _terminate_capture_session(
    process: subprocess.Popen[Any],
    grace_seconds: float,
) -> None:
    if process.poll() is not None:
        return
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        process.wait()
        return
    try:
        process.wait(timeout=grace_seconds)
        return
    except subprocess.TimeoutExpired:
        pass
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    process.wait()


def _run_capture_child(
    command: Sequence[str],
    *,
    working_directory: Path,
    stdout: Any,
    stderr: Any,
    timeout_seconds: float,
    termination_grace_seconds: float,
) -> int:
    process = subprocess.Popen(
        list(command),
        cwd=working_directory,
        stdin=subprocess.DEVNULL,
        stdout=stdout,
        stderr=stderr,
        close_fds=True,
        preexec_fn=_limit_capture_files,
        start_new_session=True,
    )
    try:
        return process.wait(timeout=timeout_seconds)
    except subprocess.TimeoutExpired as exc:
        _terminate_capture_session(process, termination_grace_seconds)
        raise CaptureError(
            f"vLLM runtime manifest capture exceeded {timeout_seconds} seconds"
        ) from exc
    except BaseException:
        _terminate_capture_session(process, termination_grace_seconds)
        raise


def _thermal_integer(value: Any, label: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise CaptureError(f"{label} must be an integer at or above {minimum}")
    return value


def validate_wsl2_thermal_stderr(
    stderr_payload: bytes,
    supervision: Wsl2ThermalSupervision,
) -> dict[str, Any]:
    events: list[dict[str, Any]] = []
    for raw_line in stderr_payload.decode("utf-8", errors="replace").splitlines():
        if not raw_line.startswith(WSL_THERMAL_EVENT_PREFIX):
            continue
        payload = raw_line.removeprefix(WSL_THERMAL_EVENT_PREFIX).encode("utf-8")
        try:
            event = benchmark.strict_json_loads(payload)
        except Exception as exc:
            raise CaptureError(f"WSL2 thermal event is not strict JSON: {exc}") from exc
        if not isinstance(event, dict):
            raise CaptureError("WSL2 thermal event must be an object")
        events.append(event)
    if [event.get("event") for event in events] != ["preflight", "complete"]:
        raise CaptureError(
            "WSL2 thermal supervision must emit exactly preflight then complete"
        )

    preflight, complete = events
    if set(preflight) != WSL_THERMAL_PREFLIGHT_KEYS:
        raise CaptureError("WSL2 preflight event fields are invalid")
    if set(complete) != WSL_THERMAL_COMPLETE_KEYS:
        raise CaptureError("WSL2 complete event fields are invalid")
    policy = supervision.policy
    for index, event in enumerate(events):
        label = ("preflight", "complete")[index]
        if event.get("schema") != WSL_THERMAL_EVENT_SCHEMA:
            raise CaptureError(f"WSL2 {label} event schema is invalid")
        if (
            event.get("policy_id") != policy.policy_id
            or event.get("policy_sha256") != policy.content_sha256
        ):
            raise CaptureError(f"WSL2 {label} event policy identity is invalid")
    if (
        preflight.get("host_limit_millicelsius")
        != policy.host_limit_millicelsius
        or preflight.get("gpu_limit_millicelsius")
        != policy.gpu_limit_millicelsius
    ):
        raise CaptureError("WSL2 preflight hard limits do not match the policy")
    if complete.get("supervision_outcome") != "child_exit":
        raise CaptureError("WSL2 thermal supervision did not report child_exit")
    if complete.get("failure_reason") is not None:
        raise CaptureError("WSL2 thermal supervision reported a failure")
    if complete.get("child_returncode") != 0:
        raise CaptureError("WSL2 thermal supervision child did not exit zero")

    values = {
        field: _thermal_integer(
            complete.get(field),
            f"WSL2 complete {field}",
            minimum=1,
        )
        for field in (
            "sample_count",
            "starting_host_millicelsius",
            "starting_gpu_millicelsius",
            "peak_host_millicelsius",
            "peak_gpu_millicelsius",
            "ending_host_millicelsius",
            "ending_gpu_millicelsius",
            "safe_handoff_stable_samples",
        )
    }
    if (
        values["starting_host_millicelsius"]
        != preflight.get("host_millicelsius")
        or values["starting_gpu_millicelsius"]
        != preflight.get("gpu_millicelsius")
    ):
        raise CaptureError("WSL2 preflight and complete starting samples disagree")
    if (
        values["peak_host_millicelsius"] >= policy.host_limit_millicelsius
        or values["peak_gpu_millicelsius"] >= policy.gpu_limit_millicelsius
    ):
        raise CaptureError("WSL2 thermal peak reached a hard limit")
    if (
        values["ending_host_millicelsius"] > policy.handoff_host_millicelsius
        or values["ending_gpu_millicelsius"] > policy.handoff_gpu_millicelsius
    ):
        raise CaptureError("WSL2 thermal safe handoff targets were not reached")
    if (
        values["safe_handoff_stable_samples"] != policy.handoff_stable_samples
        or values["sample_count"] < policy.handoff_stable_samples + 1
    ):
        raise CaptureError("WSL2 thermal stable-handoff sample evidence is invalid")
    for sensor in ("host", "gpu"):
        starting = values[f"starting_{sensor}_millicelsius"]
        peak = values[f"peak_{sensor}_millicelsius"]
        ending = values[f"ending_{sensor}_millicelsius"]
        if peak < max(starting, ending):
            raise CaptureError(f"WSL2 {sensor} peak is below an endpoint")
    return {
        "mechanism": "per-capture-windows-thermal-zone-nvml-v1",
        "policy_path": supervision.repository_path,
        "policy_id": policy.policy_id,
        "policy_sha256": policy.content_sha256,
        **values,
    }


def capture_once(
    config: Any,
    *,
    timeout_seconds: float,
    wsl2_thermal: Wsl2ThermalSupervision | None = None,
) -> Capture:
    """Execute one bounded manifest-only child and validate its exact output."""

    command = supervised_manifest_command(config, wsl2_thermal)
    termination_grace_seconds = CAPTURE_TERMINATION_GRACE_SECONDS
    if wsl2_thermal is not None:
        termination_grace_seconds += wsl2_thermal.policy.handoff_timeout_seconds
    with tempfile.TemporaryDirectory(prefix="kiln-vllm-manifest-") as directory:
        capture_root = Path(directory)
        stdout_path = capture_root / "stdout"
        stderr_path = capture_root / "stderr"
        with stdout_path.open("xb") as stdout, stderr_path.open("xb") as stderr:
            returncode = _run_capture_child(
                command,
                working_directory=config.working_directory,
                stdout=stdout,
                stderr=stderr,
                timeout_seconds=timeout_seconds,
                termination_grace_seconds=termination_grace_seconds,
            )
        stderr_bytes = stderr_path.stat().st_size
        stderr_payload = stderr_path.read_bytes()
        if returncode != 0:
            details = stderr_payload.decode("utf-8", errors="replace")[-4096:].strip()
            raise CaptureError(
                f"vLLM runtime manifest child exited {returncode}: {details}"
            )
        payload = stdout_path.read_bytes()
    if not payload or len(payload) > MAX_MANIFEST_BYTES:
        raise CaptureError(
            f"vLLM runtime manifest output must be in 1..={MAX_MANIFEST_BYTES} bytes"
        )
    try:
        value = benchmark.strict_json_loads(payload)
        manifest = benchmark.validate_vllm_runtime_manifest(
            value,
            "captured vLLM runtime manifest",
        )
        benchmark.validate_vllm_owned_launch(config, manifest)
    except Exception as exc:
        raise CaptureError(f"captured vLLM runtime manifest is invalid: {exc}") from exc
    thermal_evidence = (
        None
        if wsl2_thermal is None
        else validate_wsl2_thermal_stderr(stderr_payload, wsl2_thermal)
    )
    return Capture(
        payload=payload,
        manifest=manifest,
        stderr_bytes=stderr_bytes,
        stderr_sha256="sha256:" + hashlib.sha256(stderr_payload).hexdigest(),
        wsl2_thermal=thermal_evidence,
    )


def capture_twice(
    config: Any,
    *,
    timeout_seconds: float,
    wsl2_thermal: Wsl2ThermalSupervision | None = None,
) -> tuple[Capture, Capture]:
    """Require two byte-identical runtime and accelerator observations."""

    first = capture_once(
        config,
        timeout_seconds=timeout_seconds,
        wsl2_thermal=wsl2_thermal,
    )
    second = capture_once(
        config,
        timeout_seconds=timeout_seconds,
        wsl2_thermal=wsl2_thermal,
    )
    if first.payload != second.payload:
        raise CaptureError(
            "two vLLM runtime manifest captures were not byte-identical: "
            f"sha256:{hashlib.sha256(first.payload).hexdigest()} != "
            f"sha256:{hashlib.sha256(second.payload).hexdigest()}"
        )
    return first, second


def publish_no_clobber(path: Path, payload: bytes) -> None:
    """Durably publish the exact repeated bytes without replacing any path."""

    if path.exists() or path.is_symlink():
        raise CaptureError(f"refusing to replace existing runtime manifest: {path}")
    if not path.parent.is_dir():
        raise CaptureError(
            f"runtime manifest output parent is not a directory: {path.parent}"
        )
    temporary_path: Path | None = None
    try:
        descriptor, temporary = tempfile.mkstemp(
            prefix=f".{path.name}.",
            suffix=".tmp",
            dir=path.parent,
        )
        temporary_path = Path(temporary)
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            os.fchmod(handle.fileno(), 0o644)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary_path, path, follow_symlinks=False)
        except FileExistsError as exc:
            raise CaptureError(
                f"refusing to replace existing runtime manifest: {path}"
            ) from exc
        directory_descriptor = os.open(
            path.parent,
            os.O_RDONLY | getattr(os, "O_CLOEXEC", 0),
        )
        try:
            os.fsync(directory_descriptor)
        finally:
            os.close(directory_descriptor)
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--server-launch-config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--wsl2-thermal-policy",
        type=Path,
        help=(
            "content-hashed repository policy used to supervise and cool each "
            "capture independently"
        ),
    )
    parser.add_argument(
        "--timeout-seconds",
        type=float,
        default=DEFAULT_TIMEOUT_SECONDS,
        help=f"deadline for each of two captures (default: {DEFAULT_TIMEOUT_SECONDS:g})",
    )
    args = parser.parse_args(argv)
    if not 1.0 <= args.timeout_seconds <= 7200.0:
        parser.error("timeout-seconds must be in 1..=7200")
    return args


def main(argv: Sequence[str] | None = None) -> int:
    try:
        args = parse_args(argv)
        if args.output.exists() or args.output.is_symlink():
            raise CaptureError(
                f"refusing to replace existing runtime manifest: {args.output}"
            )
        commit = require_clean_repository()
        launch_path, _launch_repository_path = _repository_regular_file(
            args.server_launch_config,
            "server launch config",
        )
        config = benchmark.load_server_launch_config(launch_path)
        wsl2_thermal = load_platform_thermal_supervision(args.wsl2_thermal_policy)
        first, second = capture_twice(
            config,
            timeout_seconds=args.timeout_seconds,
            wsl2_thermal=wsl2_thermal,
        )
        if require_clean_repository() != commit:
            raise CaptureError(
                "repository commit changed during vLLM manifest capture"
            )
        publish_no_clobber(args.output, first.payload)
        result = {
            "capture_count": 2,
            "manifest_bytes": len(first.payload),
            "manifest_sha256": "sha256:" + hashlib.sha256(first.payload).hexdigest(),
            "output": str(args.output.absolute()),
            "runtime_content_sha256": first.manifest["runtime_content_sha256"],
            "source_commit": commit,
            "stderr": [
                {
                    "bytes": first.stderr_bytes,
                    "sha256": first.stderr_sha256,
                },
                {
                    "bytes": second.stderr_bytes,
                    "sha256": second.stderr_sha256,
                },
            ],
            "system_fingerprint": first.manifest["system_fingerprint"],
            "wsl2_thermal_supervision": (
                None
                if wsl2_thermal is None
                else [first.wsl2_thermal, second.wsl2_thermal]
            ),
        }
        print(
            json.dumps(
                result,
                allow_nan=False,
                ensure_ascii=True,
                separators=(",", ":"),
                sort_keys=True,
            )
        )
        return 0
    except (CaptureError, benchmark.BenchmarkError, OSError, subprocess.SubprocessError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
