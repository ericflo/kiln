#!/usr/bin/env python3
"""Capture a repeatable vLLM runtime manifest from an owned-launch document."""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import importlib.util
import json
import os
import resource
import signal
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


class CaptureError(RuntimeError):
    """Raised when a runtime manifest cannot be captured reproducibly."""


@dataclasses.dataclass(frozen=True)
class Capture:
    payload: bytes
    manifest: dict[str, Any]
    stderr_bytes: int
    stderr_sha256: str


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


def require_committed_file(path: Path, label: str) -> Path:
    """Require a regular repository file whose bytes match HEAD."""

    candidate = path if path.is_absolute() else ROOT / path
    absolute = Path(os.path.abspath(candidate))
    try:
        relative = absolute.relative_to(ROOT)
    except ValueError as exc:
        raise CaptureError(f"{label} must stay inside {ROOT}") from exc
    current = ROOT
    for part in relative.parts:
        current /= part
        try:
            metadata = current.lstat()
        except OSError as exc:
            raise CaptureError(f"{label} is unavailable: {exc}") from exc
        if current.is_symlink():
            raise CaptureError(f"{label} must not traverse a symlink")
    if not absolute.is_file():
        raise CaptureError(f"{label} must be a regular, non-symlink file")
    committed = subprocess.run(
        ["git", "show", f"HEAD:{relative.as_posix()}"],
        cwd=ROOT,
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    if committed.returncode != 0:
        raise CaptureError(f"{label} must be tracked by the current commit")
    if committed.stdout != absolute.read_bytes():
        raise CaptureError(f"{label} bytes do not match the current commit")
    return absolute


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


def _limit_capture_files() -> None:
    resource.setrlimit(resource.RLIMIT_FSIZE, (MAX_STDERR_BYTES, MAX_STDERR_BYTES))


def _terminate_capture_session(
    process: subprocess.Popen[Any],
    termination_grace_seconds: float,
) -> None:
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        pass
    try:
        process.wait(timeout=termination_grace_seconds)
        return
    except subprocess.TimeoutExpired:
        pass
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    process.wait(timeout=termination_grace_seconds)


def _run_capture_child(
    command: list[str],
    *,
    working_directory: Path,
    stdout: Any,
    stderr: Any,
    timeout_seconds: float,
    termination_grace_seconds: float = 30.0,
) -> subprocess.CompletedProcess[bytes]:
    process = subprocess.Popen(
        command,
        cwd=working_directory,
        stdin=subprocess.DEVNULL,
        stdout=stdout,
        stderr=stderr,
        close_fds=True,
        preexec_fn=_limit_capture_files,
        start_new_session=True,
    )
    try:
        returncode = process.wait(timeout=timeout_seconds)
    except subprocess.TimeoutExpired as exc:
        _terminate_capture_session(process, termination_grace_seconds)
        raise CaptureError(
            f"vLLM runtime manifest capture exceeded {timeout_seconds} seconds"
        ) from exc
    return subprocess.CompletedProcess(command, returncode)


def capture_once(config: Any, *, timeout_seconds: float) -> Capture:
    """Execute one bounded manifest-only child and validate its exact output."""

    command = manifest_command(config)
    with tempfile.TemporaryDirectory(prefix="kiln-vllm-manifest-") as directory:
        capture_root = Path(directory)
        stdout_path = capture_root / "stdout"
        stderr_path = capture_root / "stderr"
        with stdout_path.open("xb") as stdout, stderr_path.open("xb") as stderr:
            completed = _run_capture_child(
                command,
                working_directory=config.working_directory,
                stdout=stdout,
                stderr=stderr,
                timeout_seconds=timeout_seconds,
            )
        stderr_bytes = stderr_path.stat().st_size
        stderr_payload = stderr_path.read_bytes()
        if completed.returncode != 0:
            details = stderr_payload.decode("utf-8", errors="replace")[-4096:].strip()
            raise CaptureError(
                f"vLLM runtime manifest child exited {completed.returncode}: {details}"
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
    return Capture(
        payload=payload,
        manifest=manifest,
        stderr_bytes=stderr_bytes,
        stderr_sha256="sha256:" + hashlib.sha256(stderr_payload).hexdigest(),
    )


def capture_twice(config: Any, *, timeout_seconds: float) -> tuple[Capture, Capture]:
    """Require two byte-identical runtime and accelerator observations."""

    first = capture_once(config, timeout_seconds=timeout_seconds)
    second = capture_once(config, timeout_seconds=timeout_seconds)
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
        launch_path = require_committed_file(
            args.server_launch_config,
            "server launch config",
        )
        config = benchmark.load_server_launch_config(launch_path)
        first, second = capture_twice(
            config,
            timeout_seconds=args.timeout_seconds,
        )
        if require_clean_repository() != commit:
            raise CaptureError("repository commit changed during manifest capture")
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
