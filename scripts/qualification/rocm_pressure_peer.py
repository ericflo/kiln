#!/usr/bin/env python3
"""Hold bounded external HIP allocations at a target DRM free fraction."""

from __future__ import annotations

import argparse
import ctypes
import json
import math
import os
import re
import signal
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


DRM_ROOT = Path("/sys/class/drm")
HIP_LIBRARY = Path("/opt/rocm/lib/libamdhip64.so")
CARD_RE = re.compile(r"^card[0-9]+$")
MIB = 1024 * 1024
GIB = 1024 * MIB
MIN_ALLOCATION_BYTES = 64 * MIB
ALLOCATION_ALIGNMENT_BYTES = 2 * MIB


class PressurePeerError(RuntimeError):
    pass


@dataclass(frozen=True)
class DrmMemorySnapshot:
    total_bytes: int
    used_bytes: int
    free_bytes: int
    free_fraction: float


def _read_nonnegative_integer(path: Path) -> int | None:
    try:
        value = int(path.read_text(encoding="utf-8").strip())
    except (OSError, UnicodeError, ValueError):
        return None
    return value if value >= 0 else None


def _max_field(root: Path, field: str) -> int:
    values: list[int] = []
    try:
        entries = list(root.iterdir())
    except OSError as exc:
        raise PressurePeerError(f"cannot inspect DRM root {root}: {exc}") from exc
    for entry in entries:
        if not CARD_RE.fullmatch(entry.name):
            continue
        value = _read_nonnegative_integer(entry / "device" / field)
        if value is not None:
            values.append(value)
    return max(values, default=0)


def read_drm_memory_snapshot(root: Path = DRM_ROOT) -> DrmMemorySnapshot:
    total = _max_field(root, "mem_info_vram_total") + _max_field(
        root, "mem_info_gtt_total"
    )
    if total <= 0:
        raise PressurePeerError("DRM exposes no positive VRAM/GTT total")
    used = min(
        total,
        _max_field(root, "mem_info_vram_used")
        + _max_field(root, "mem_info_gtt_used"),
    )
    free = total - used
    return DrmMemorySnapshot(total, used, free, free / total)


def next_allocation_bytes(
    snapshot: DrmMemorySnapshot,
    *,
    target_free_fraction: float,
    minimum_free_fraction: float,
    chunk_bytes: int,
    remaining_budget_bytes: int,
) -> int:
    target_free = math.ceil(snapshot.total_bytes * target_free_fraction)
    minimum_free = math.ceil(snapshot.total_bytes * minimum_free_fraction)
    needed = snapshot.free_bytes - target_free
    safe = snapshot.free_bytes - minimum_free
    size = min(needed, safe, chunk_bytes, remaining_budget_bytes)
    if size <= 0:
        return 0
    return (size // ALLOCATION_ALIGNMENT_BYTES) * ALLOCATION_ALIGNMENT_BYTES


def require_minimum_free(
    snapshot: DrmMemorySnapshot, minimum_free_fraction: float
) -> None:
    if snapshot.free_fraction < minimum_free_fraction:
        raise PressurePeerError(
            "observed free fraction crossed the safety floor: "
            f"{snapshot.free_fraction:.6f} < {minimum_free_fraction:.6f}"
        )


class HipRuntime:
    def __init__(self, path: Path) -> None:
        try:
            self.library = ctypes.CDLL(str(path))
        except OSError as exc:
            raise PressurePeerError(f"cannot load HIP runtime {path}: {exc}") from exc
        self.library.hipSetDevice.argtypes = [ctypes.c_int]
        self.library.hipSetDevice.restype = ctypes.c_int
        self.library.hipMalloc.argtypes = [
            ctypes.POINTER(ctypes.c_void_p),
            ctypes.c_size_t,
        ]
        self.library.hipMalloc.restype = ctypes.c_int
        self.library.hipFree.argtypes = [ctypes.c_void_p]
        self.library.hipFree.restype = ctypes.c_int

    def set_device(self, ordinal: int) -> None:
        self._check(self.library.hipSetDevice(ordinal), "hipSetDevice")

    def malloc(self, size: int) -> ctypes.c_void_p:
        pointer = ctypes.c_void_p()
        self._check(self.library.hipMalloc(ctypes.byref(pointer), size), "hipMalloc")
        if pointer.value is None:
            raise PressurePeerError("hipMalloc succeeded with a null pointer")
        return pointer

    def free(self, pointer: ctypes.c_void_p) -> None:
        self._check(self.library.hipFree(pointer), "hipFree")

    @staticmethod
    def _check(code: int, operation: str) -> None:
        if code != 0:
            raise PressurePeerError(f"{operation} failed with HIP status {code}")


def emit(event: str, **fields: Any) -> None:
    print(
        json.dumps({"event": event, **fields}, sort_keys=True, separators=(",", ":")),
        flush=True,
    )


def write_ready(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    encoded = (json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n").encode()
    try:
        with temporary.open("xb") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def validate_args(args: argparse.Namespace) -> None:
    if not 0.05 <= args.minimum_free_fraction < args.target_free_fraction < 0.10:
        raise PressurePeerError(
            "fractions must satisfy 0.05 <= minimum < target < 0.10"
        )
    if args.chunk_mib < 64 or args.chunk_mib > 2048:
        raise PressurePeerError("--chunk-mib must be between 64 and 2048")
    if args.max_allocation_gib <= 0 or args.max_allocation_gib > 128:
        raise PressurePeerError("--max-allocation-gib must be in (0, 128]")
    if args.hold_seconds <= 0 or args.hold_seconds > 900:
        raise PressurePeerError("--hold-seconds must be in (0, 900]")


def run(args: argparse.Namespace) -> None:
    validate_args(args)
    ready_path = args.ready_file.resolve(strict=False)
    if ready_path.exists():
        raise PressurePeerError(f"ready file already exists: {ready_path}")

    stop = False

    def request_stop(_signum: int, _frame: Any) -> None:
        nonlocal stop
        stop = True

    signal.signal(signal.SIGINT, request_stop)
    signal.signal(signal.SIGTERM, request_stop)

    hip = HipRuntime(args.hip_library)
    hip.set_device(args.device)
    baseline = read_drm_memory_snapshot(args.drm_root)
    pointers: list[ctypes.c_void_p] = []
    allocated_bytes = 0
    max_allocation_bytes = int(args.max_allocation_gib * GIB)
    chunk_bytes = args.chunk_mib * MIB
    started = time.monotonic()
    try:
        while not stop:
            snapshot = read_drm_memory_snapshot(args.drm_root)
            require_minimum_free(snapshot, args.minimum_free_fraction)
            if snapshot.free_fraction <= args.target_free_fraction:
                payload = {
                    "schema_version": 1,
                    "pid": os.getpid(),
                    "allocated_bytes": allocated_bytes,
                    "target_free_fraction": args.target_free_fraction,
                    "minimum_free_fraction": args.minimum_free_fraction,
                    "baseline": asdict(baseline),
                    "ready": asdict(snapshot),
                }
                write_ready(ready_path, payload)
                emit("pressure_ready", **payload)
                deadline = time.monotonic() + args.hold_seconds
                while not stop and time.monotonic() < deadline:
                    time.sleep(0.25)
                    require_minimum_free(
                        read_drm_memory_snapshot(args.drm_root),
                        args.minimum_free_fraction,
                    )
                return

            size = next_allocation_bytes(
                snapshot,
                target_free_fraction=args.target_free_fraction,
                minimum_free_fraction=args.minimum_free_fraction,
                chunk_bytes=chunk_bytes,
                remaining_budget_bytes=max_allocation_bytes - allocated_bytes,
            )
            if size < MIN_ALLOCATION_BYTES:
                raise PressurePeerError(
                    "cannot reach target without crossing the minimum free fraction "
                    "or maximum allocation budget"
                )
            pointers.append(hip.malloc(size))
            allocated_bytes += size
            if len(pointers) == 1 or len(pointers) % 8 == 0:
                emit(
                    "pressure_allocation",
                    allocations=len(pointers),
                    allocated_bytes=allocated_bytes,
                    observed_free_bytes=snapshot.free_bytes,
                )
            time.sleep(0.05)
        raise PressurePeerError("stopped before reaching the target free fraction")
    finally:
        failures: list[str] = []
        for pointer in reversed(pointers):
            try:
                hip.free(pointer)
            except PressurePeerError as exc:
                failures.append(str(exc))
        emit(
            "pressure_released",
            allocated_bytes=allocated_bytes,
            allocation_count=len(pointers),
            elapsed_seconds=time.monotonic() - started,
            free_failures=failures[:8],
        )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ready-file", required=True, type=Path)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--target-free-fraction", type=float, default=0.08)
    parser.add_argument("--minimum-free-fraction", type=float, default=0.05)
    parser.add_argument("--chunk-mib", type=int, default=512)
    parser.add_argument("--max-allocation-gib", type=float, default=110.0)
    parser.add_argument("--hold-seconds", type=float, default=300.0)
    parser.add_argument("--drm-root", type=Path, default=DRM_ROOT)
    parser.add_argument("--hip-library", type=Path, default=HIP_LIBRARY)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    try:
        run(parse_args(argv))
        return 0
    except Exception as exc:
        emit("pressure_error", error=f"{type(exc).__name__}: {exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
