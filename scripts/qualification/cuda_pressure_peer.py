#!/usr/bin/env python3
"""Hold bounded external CUDA allocations above a hard free-memory floor."""

from __future__ import annotations

import argparse
import ctypes
import json
import os
import signal
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


CUDA_LIBRARY = Path("/usr/lib/wsl/lib/libcuda.so.1")
NVIDIA_SMI = Path("/usr/lib/wsl/lib/nvidia-smi")
MIB = 1024 * 1024
GIB = 1024 * MIB
MINIMUM_ALLOWED_FREE_MIB = 768
MIN_ALLOCATION_BYTES = 64 * MIB
ALLOCATION_ALIGNMENT_BYTES = 2 * MIB


class PressurePeerError(RuntimeError):
    pass


@dataclass(frozen=True)
class CudaMemorySnapshot:
    total_bytes: int
    free_bytes: int


def parse_nvidia_memory_snapshot(output: str) -> CudaMemorySnapshot:
    lines = [line.strip() for line in output.splitlines() if line.strip()]
    if len(lines) != 1:
        raise PressurePeerError(
            "nvidia-smi memory query returned an unexpected number of rows"
        )
    fields = [field.strip() for field in lines[0].split(",")]
    if len(fields) != 2:
        raise PressurePeerError(
            "nvidia-smi memory query returned an unexpected shape"
        )
    try:
        total_mib, free_mib = (int(field) for field in fields)
    except ValueError as exc:
        raise PressurePeerError(
            "nvidia-smi memory query returned a non-integer value"
        ) from exc
    if total_mib <= 0 or free_mib < 0 or free_mib > total_mib:
        raise PressurePeerError(
            "nvidia-smi memory query returned invalid totals"
        )
    return CudaMemorySnapshot(total_mib * MIB, free_mib * MIB)


def nvidia_memory_snapshot(
    nvidia_smi: Path, device_ordinal: int
) -> CudaMemorySnapshot:
    try:
        completed = subprocess.run(
            [
                str(nvidia_smi),
                "-i",
                str(device_ordinal),
                "--query-gpu=memory.total,memory.free",
                "--format=csv,noheader,nounits",
            ],
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=10.0,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise PressurePeerError(f"nvidia-smi memory query failed: {exc}") from exc
    if completed.returncode != 0:
        raise PressurePeerError(
            "nvidia-smi memory query exited "
            f"{completed.returncode}: {completed.stderr[-1000:]}"
        )
    return parse_nvidia_memory_snapshot(completed.stdout)


def next_allocation_bytes(
    snapshot: CudaMemorySnapshot,
    *,
    target_free_bytes: int,
    minimum_free_bytes: int,
    chunk_bytes: int,
    remaining_budget_bytes: int,
) -> int:
    needed = snapshot.free_bytes - target_free_bytes
    safe = snapshot.free_bytes - minimum_free_bytes
    size = min(needed, safe, chunk_bytes, remaining_budget_bytes)
    if size <= 0:
        return 0
    if size < MIN_ALLOCATION_BYTES:
        if min(safe, chunk_bytes, remaining_budget_bytes) < MIN_ALLOCATION_BYTES:
            return 0
        size = MIN_ALLOCATION_BYTES
    return (size // ALLOCATION_ALIGNMENT_BYTES) * ALLOCATION_ALIGNMENT_BYTES


def require_minimum_free(
    snapshot: CudaMemorySnapshot, minimum_free_bytes: int
) -> None:
    if snapshot.free_bytes < minimum_free_bytes:
        raise PressurePeerError(
            "observed CUDA free memory crossed the safety floor: "
            f"{snapshot.free_bytes} < {minimum_free_bytes} bytes"
        )


class CudaDriver:
    def __init__(self, path: Path, device_ordinal: int) -> None:
        try:
            self.library = ctypes.CDLL(str(path))
        except OSError as exc:
            raise PressurePeerError(f"cannot load CUDA driver {path}: {exc}") from exc

        self.library.cuInit.argtypes = [ctypes.c_uint]
        self.library.cuInit.restype = ctypes.c_int
        self.library.cuDeviceGet.argtypes = [
            ctypes.POINTER(ctypes.c_int),
            ctypes.c_int,
        ]
        self.library.cuDeviceGet.restype = ctypes.c_int
        self.library.cuDevicePrimaryCtxRetain.argtypes = [
            ctypes.POINTER(ctypes.c_void_p),
            ctypes.c_int,
        ]
        self.library.cuDevicePrimaryCtxRetain.restype = ctypes.c_int
        self.library.cuDevicePrimaryCtxRelease.argtypes = [ctypes.c_int]
        self.library.cuDevicePrimaryCtxRelease.restype = ctypes.c_int
        self.library.cuCtxSetCurrent.argtypes = [ctypes.c_void_p]
        self.library.cuCtxSetCurrent.restype = ctypes.c_int
        self.library.cuMemGetInfo_v2.argtypes = [
            ctypes.POINTER(ctypes.c_size_t),
            ctypes.POINTER(ctypes.c_size_t),
        ]
        self.library.cuMemGetInfo_v2.restype = ctypes.c_int
        self.library.cuMemAlloc_v2.argtypes = [
            ctypes.POINTER(ctypes.c_uint64),
            ctypes.c_size_t,
        ]
        self.library.cuMemAlloc_v2.restype = ctypes.c_int
        self.library.cuMemFree_v2.argtypes = [ctypes.c_uint64]
        self.library.cuMemFree_v2.restype = ctypes.c_int
        self.library.cuGetErrorName.argtypes = [
            ctypes.c_int,
            ctypes.POINTER(ctypes.c_char_p),
        ]
        self.library.cuGetErrorName.restype = ctypes.c_int

        self._check(self.library.cuInit(0), "cuInit")
        device = ctypes.c_int()
        self._check(
            self.library.cuDeviceGet(ctypes.byref(device), device_ordinal),
            "cuDeviceGet",
        )
        self.device = device.value
        context = ctypes.c_void_p()
        self._check(
            self.library.cuDevicePrimaryCtxRetain(
                ctypes.byref(context), self.device
            ),
            "cuDevicePrimaryCtxRetain",
        )
        if context.value is None:
            raise PressurePeerError(
                "cuDevicePrimaryCtxRetain succeeded with a null context"
            )
        self.context = context
        self.released = False
        try:
            self._check(
                self.library.cuCtxSetCurrent(self.context), "cuCtxSetCurrent"
            )
        except Exception:
            self.release()
            raise

    def snapshot(self) -> CudaMemorySnapshot:
        free = ctypes.c_size_t()
        total = ctypes.c_size_t()
        self._check(
            self.library.cuMemGetInfo_v2(
                ctypes.byref(free), ctypes.byref(total)
            ),
            "cuMemGetInfo_v2",
        )
        if total.value <= 0 or free.value > total.value:
            raise PressurePeerError(
                "CUDA driver returned an invalid memory snapshot: "
                f"free={free.value}, total={total.value}"
            )
        return CudaMemorySnapshot(total.value, free.value)

    def allocate(self, size: int) -> int:
        pointer = ctypes.c_uint64()
        self._check(
            self.library.cuMemAlloc_v2(ctypes.byref(pointer), size),
            "cuMemAlloc_v2",
        )
        if pointer.value == 0:
            raise PressurePeerError(
                "cuMemAlloc_v2 succeeded with a null device pointer"
            )
        return pointer.value

    def free(self, pointer: int) -> None:
        self._check(self.library.cuMemFree_v2(pointer), "cuMemFree_v2")

    def release(self) -> None:
        if self.released:
            return
        self.released = True
        self._check(
            self.library.cuDevicePrimaryCtxRelease(self.device),
            "cuDevicePrimaryCtxRelease",
        )

    def _check(self, code: int, operation: str) -> None:
        if code == 0:
            return
        name = ctypes.c_char_p()
        error_name = f"CUDA status {code}"
        if self.library.cuGetErrorName(code, ctypes.byref(name)) == 0 and name.value:
            error_name = name.value.decode("ascii", errors="replace")
        raise PressurePeerError(f"{operation} failed with {error_name} ({code})")


def emit(event: str, **fields: Any) -> None:
    print(
        json.dumps(
            {"event": event, **fields},
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
        ),
        flush=True,
    )


def write_json_no_clobber(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    encoded = (
        json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        )
        + "\n"
    ).encode("ascii")
    try:
        with temporary.open("xb") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def validate_args(args: argparse.Namespace) -> None:
    for name in (
        "target_free_mib",
        "minimum_free_mib",
        "chunk_mib",
        "max_allocation_mib",
    ):
        value = getattr(args, name)
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise PressurePeerError(f"--{name.replace('_', '-')} must be positive")
    if args.minimum_free_mib < MINIMUM_ALLOWED_FREE_MIB:
        raise PressurePeerError(
            "--minimum-free-mib must be at least "
            f"{MINIMUM_ALLOWED_FREE_MIB}"
        )
    if args.target_free_mib < args.minimum_free_mib + 256:
        raise PressurePeerError(
            "--target-free-mib must leave at least 256 MiB above the floor"
        )
    if not 64 <= args.chunk_mib <= 1024:
        raise PressurePeerError("--chunk-mib must be in 64..=1024")
    if args.max_allocation_mib > 8192:
        raise PressurePeerError("--max-allocation-mib must not exceed 8192")
    if args.nvidia_smi != NVIDIA_SMI:
        raise PressurePeerError(
            f"--nvidia-smi must be the reviewed WSL2 binary {NVIDIA_SMI}"
        )
    if (
        not isinstance(args.hold_seconds, float)
        or not 1.0 <= args.hold_seconds <= 900.0
    ):
        raise PressurePeerError("--hold-seconds must be in 1..=900")
    if (
        not isinstance(args.poll_milliseconds, int)
        or not 50 <= args.poll_milliseconds <= 1000
    ):
        raise PressurePeerError("--poll-milliseconds must be in 50..=1000")


def run(args: argparse.Namespace) -> None:
    validate_args(args)
    ready_path = args.ready_file.resolve(strict=False)
    release_path = args.release_file.resolve(strict=False)
    if ready_path == release_path:
        raise PressurePeerError("ready and release files must be distinct")
    if ready_path.exists() or release_path.exists():
        raise PressurePeerError("ready or release file already exists")

    stop = False

    def request_stop(_signum: int, _frame: Any) -> None:
        nonlocal stop
        stop = True

    signal.signal(signal.SIGINT, request_stop)
    signal.signal(signal.SIGTERM, request_stop)

    target_free_bytes = args.target_free_mib * MIB
    minimum_free_bytes = args.minimum_free_mib * MIB
    chunk_bytes = args.chunk_mib * MIB
    max_allocation_bytes = args.max_allocation_mib * MIB
    driver = CudaDriver(args.cuda_library, args.device)
    pointers: list[int] = []
    allocated_bytes = 0
    ready_written = False
    baseline: CudaMemorySnapshot | None = None
    allocator_baseline: CudaMemorySnapshot | None = None
    allocator_ready: CudaMemorySnapshot | None = None
    minimum_observed_free_bytes: int | None = None
    sample_count = 0
    started = time.monotonic()
    failure: BaseException | None = None
    release_failures: list[str] = []
    final_snapshot: CudaMemorySnapshot | None = None
    allocator_final_snapshot: CudaMemorySnapshot | None = None
    try:
        baseline = nvidia_memory_snapshot(args.nvidia_smi, args.device)
        allocator_baseline = driver.snapshot()
        require_minimum_free(baseline, minimum_free_bytes)
        effective_target_free_bytes = min(
            target_free_bytes,
            baseline.free_bytes - MIN_ALLOCATION_BYTES,
        )
        if effective_target_free_bytes < minimum_free_bytes:
            raise PressurePeerError(
                "global CUDA free memory cannot retain both the qualifying "
                "allocation and safety floor"
            )
        while not stop:
            snapshot = nvidia_memory_snapshot(args.nvidia_smi, args.device)
            sample_count += 1
            minimum_observed_free_bytes = min(
                snapshot.free_bytes,
                (
                    snapshot.free_bytes
                    if minimum_observed_free_bytes is None
                    else minimum_observed_free_bytes
                ),
            )
            require_minimum_free(snapshot, minimum_free_bytes)
            if snapshot.free_bytes <= effective_target_free_bytes:
                if allocated_bytes < MIN_ALLOCATION_BYTES:
                    raise PressurePeerError(
                        "pressure target was reached without a qualifying external "
                        f"allocation of at least {MIN_ALLOCATION_BYTES} bytes"
                    )
                payload = {
                    "schema_version": 2,
                    "pid": os.getpid(),
                    "device_ordinal": args.device,
                    "memory_source": "nvidia-smi",
                    "allocator_memory_source": "cuMemGetInfo_v2",
                    "allocated_bytes": allocated_bytes,
                    "allocation_count": len(pointers),
                    "configured_target_free_bytes": target_free_bytes,
                    "effective_target_free_bytes": effective_target_free_bytes,
                    "minimum_free_bytes": minimum_free_bytes,
                    "baseline": asdict(baseline),
                    "ready": asdict(snapshot),
                    "allocator_baseline": asdict(allocator_baseline),
                    "allocator_ready": asdict(allocator_ready),
                }
                write_json_no_clobber(ready_path, payload)
                ready_written = True
                emit("cuda_pressure_ready", **payload)
                deadline = time.monotonic() + args.hold_seconds
                while not stop and time.monotonic() < deadline:
                    time.sleep(args.poll_milliseconds / 1000.0)
                    held = nvidia_memory_snapshot(
                        args.nvidia_smi, args.device
                    )
                    sample_count += 1
                    minimum_observed_free_bytes = min(
                        held.free_bytes,
                        (
                            held.free_bytes
                            if minimum_observed_free_bytes is None
                            else minimum_observed_free_bytes
                        ),
                    )
                    require_minimum_free(held, minimum_free_bytes)
                return

            size = next_allocation_bytes(
                snapshot,
                target_free_bytes=effective_target_free_bytes,
                minimum_free_bytes=minimum_free_bytes,
                chunk_bytes=chunk_bytes,
                remaining_budget_bytes=max_allocation_bytes - allocated_bytes,
            )
            if size < MIN_ALLOCATION_BYTES:
                raise PressurePeerError(
                    "cannot reach the pressure target without crossing the safety "
                    "floor or allocation budget"
                )
            pointer = driver.allocate(size)
            pointers.append(pointer)
            allocated_bytes += size
            allocator_ready = driver.snapshot()
            after = nvidia_memory_snapshot(args.nvidia_smi, args.device)
            sample_count += 1
            minimum_observed_free_bytes = min(
                after.free_bytes,
                (
                    after.free_bytes
                    if minimum_observed_free_bytes is None
                    else minimum_observed_free_bytes
                ),
            )
            require_minimum_free(after, minimum_free_bytes)
            if len(pointers) == 1 or len(pointers) % 8 == 0:
                emit(
                    "cuda_pressure_allocation",
                    allocated_bytes=allocated_bytes,
                    allocation_count=len(pointers),
                    observed_free_bytes=after.free_bytes,
                )
            time.sleep(0.05)
        raise PressurePeerError("stopped before reaching the pressure target")
    except BaseException as exc:
        failure = exc
        raise
    finally:
        for pointer in reversed(pointers):
            try:
                driver.free(pointer)
            except PressurePeerError as exc:
                release_failures.append(str(exc))
        try:
            allocator_final_snapshot = driver.snapshot()
        except PressurePeerError as exc:
            release_failures.append(str(exc))
        try:
            driver.release()
        except PressurePeerError as exc:
            release_failures.append(str(exc))
        try:
            final_snapshot = nvidia_memory_snapshot(
                args.nvidia_smi, args.device
            )
        except PressurePeerError as exc:
            release_failures.append(str(exc))
        release_payload = {
            "schema_version": 2,
            "pid": os.getpid(),
            "device_ordinal": args.device,
            "memory_source": "nvidia-smi",
            "allocator_memory_source": "cuMemGetInfo_v2",
            "ready_written": ready_written,
            "completed": failure is None and not release_failures,
            "allocated_bytes": allocated_bytes,
            "allocation_count": len(pointers),
            "minimum_observed_free_bytes": minimum_observed_free_bytes,
            "sample_count": sample_count,
            "elapsed_seconds": time.monotonic() - started,
            "release_failures": release_failures[:8],
            "final": None if final_snapshot is None else asdict(final_snapshot),
            "allocator_final": (
                None
                if allocator_final_snapshot is None
                else asdict(allocator_final_snapshot)
            ),
        }
        try:
            write_json_no_clobber(release_path, release_payload)
        except Exception as exc:
            emit("cuda_pressure_release_receipt_failed", error=str(exc))
            if failure is None:
                raise
        emit("cuda_pressure_released", **release_payload)
        if release_failures and failure is None:
            raise PressurePeerError(
                "CUDA pressure release failed: " + " | ".join(release_failures)
            )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ready-file", required=True, type=Path)
    parser.add_argument("--release-file", required=True, type=Path)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--target-free-mib", type=int, default=1024)
    parser.add_argument("--minimum-free-mib", type=int, default=768)
    parser.add_argument("--chunk-mib", type=int, default=256)
    parser.add_argument("--max-allocation-mib", type=int, default=1280)
    parser.add_argument("--hold-seconds", type=float, default=300.0)
    parser.add_argument("--poll-milliseconds", type=int, default=100)
    parser.add_argument("--cuda-library", type=Path, default=CUDA_LIBRARY)
    parser.add_argument("--nvidia-smi", type=Path, default=NVIDIA_SMI)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    try:
        run(parse_args(argv))
        return 0
    except Exception as exc:
        emit("cuda_pressure_error", error=f"{type(exc).__name__}: {exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
