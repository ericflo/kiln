"""Typed device-memory counters for serving qualification.

The benchmark samples whole-device used memory. DRM sysfs and NVML expose the
same quantity through different interfaces; callers receive a closed receipt
identity so a comparison cannot silently measure a different accelerator.
"""

from __future__ import annotations

import ctypes
import ctypes.util
import glob
import threading
from pathlib import Path
from typing import Any, Callable


class DeviceMemoryError(RuntimeError):
    """Raised when device-memory identity or sampling is unavailable."""


class NvmlUnavailable(DeviceMemoryError):
    """Raised when auto-detection finds no usable NVML runtime."""


class _NvmlMemory(ctypes.Structure):
    _fields_ = [
        ("total", ctypes.c_ulonglong),
        ("free", ctypes.c_ulonglong),
        ("used", ctypes.c_ulonglong),
    ]


def _configure(function: Any, argtypes: list[Any], restype: Any) -> Any:
    function.argtypes = argtypes
    function.restype = restype
    return function


def _nvml_function(
    library: Any,
    names: tuple[str, ...],
    argtypes: list[Any],
    restype: Any = ctypes.c_int,
) -> Any:
    for name in names:
        function = getattr(library, name, None)
        if function is not None:
            return _configure(function, argtypes, restype)
    raise DeviceMemoryError(f"NVML library lacks required symbol {names[0]}")


def _decode_buffer(buffer: ctypes.Array[ctypes.c_char], label: str) -> str:
    value = bytes(buffer).split(b"\0", 1)[0]
    try:
        decoded = value.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise DeviceMemoryError(f"NVML {label} is not UTF-8") from exc
    if not decoded or any(ord(character) < 32 for character in decoded):
        raise DeviceMemoryError(f"NVML {label} is empty or contains control bytes")
    return decoded


class DrmMemoryCounter:
    """Linux DRM whole-device used-memory counter."""

    source = "drm_vram_used"

    def __init__(self, path: Path) -> None:
        self.path = path.resolve()

    def read_bytes(self) -> int:
        try:
            value = int(self.path.read_text(encoding="utf-8").strip())
        except (OSError, ValueError) as exc:
            raise DeviceMemoryError(
                f"cannot read DRM memory counter {self.path}: {exc}"
            ) from exc
        if value < 0:
            raise DeviceMemoryError(f"DRM memory counter at {self.path} is negative")
        return value

    def receipt_identity(self) -> dict[str, Any]:
        return {
            "source": self.source,
            "path": str(self.path),
            "device": None,
        }

    def close(self) -> None:
        return None


class NvmlMemoryCounter:
    """NVML whole-device used-memory counter with stable GPU identity."""

    source = "nvml_used"

    def __init__(
        self,
        device_index: int | None,
        *,
        device_uuid: str | None = None,
        library_loader: Callable[[str], Any] = ctypes.CDLL,
        library_name: str | None = None,
    ) -> None:
        if device_index is not None and device_uuid is not None:
            raise DeviceMemoryError("NVML index and UUID selectors are mutually exclusive")
        requested_library = library_name or ctypes.util.find_library("nvidia-ml")
        if requested_library is None:
            requested_library = "libnvidia-ml.so.1"
        try:
            self._library = library_loader(requested_library)
        except OSError as exc:
            raise NvmlUnavailable(
                f"cannot load NVML library {requested_library!r}: {exc}"
            ) from exc
        self._library_name = str(getattr(self._library, "_name", requested_library))
        self._closed = True
        try:
            self._bind_functions()
            init_result = self._init()
            if init_result != 0:
                raise NvmlUnavailable(
                    f"nvmlInit_v2 failed: {self._error_detail(init_result)}"
                )
            self._closed = False
            count = ctypes.c_uint()
            self._check(self._device_count(ctypes.byref(count)), "nvmlDeviceGetCount_v2")
            if count.value == 0:
                raise NvmlUnavailable("NVML reports zero devices")
            handle: ctypes.c_void_p
            if device_uuid is not None:
                matches: list[tuple[int, ctypes.c_void_p]] = []
                for candidate_index in range(count.value):
                    candidate_handle = self._handle_at(candidate_index)
                    candidate_uuid = self._read_string_for_handle(
                        candidate_handle,
                        "nvmlDeviceGetUUID",
                        self._device_uuid,
                        96,
                    )
                    if candidate_uuid == device_uuid:
                        matches.append((candidate_index, candidate_handle))
                if len(matches) != 1:
                    raise DeviceMemoryError(
                        f"NVML UUID {device_uuid!r} matched {len(matches)} devices"
                    )
                device_index, handle = matches[0]
                selector = "explicit_uuid"
            elif device_index is None:
                if count.value != 1:
                    raise DeviceMemoryError(
                        f"NVML reports {count.value} devices; select one with "
                        "--memory-device-index or --memory-device-uuid"
                    )
                device_index = 0
                handle = self._handle_at(device_index)
                selector = "auto_single_device"
            else:
                if device_index < 0 or device_index >= count.value:
                    raise DeviceMemoryError(
                        f"NVML device index {device_index} is outside 0..{count.value - 1}"
                    )
                handle = self._handle_at(device_index)
                selector = "explicit_index"
            self._handle = handle
            self._device_index = device_index
            self._device_count_value = count.value
            self._selector = selector
            self._identity = self._read_identity()
            self.read_bytes()
        except Exception:
            self.close()
            raise

    def _bind_functions(self) -> None:
        self._init = _nvml_function(
            self._library, ("nvmlInit_v2", "nvmlInit"), []
        )
        self._shutdown = _nvml_function(self._library, ("nvmlShutdown",), [])
        self._device_count = _nvml_function(
            self._library,
            ("nvmlDeviceGetCount_v2", "nvmlDeviceGetCount"),
            [ctypes.POINTER(ctypes.c_uint)],
        )
        self._device_by_index = _nvml_function(
            self._library,
            ("nvmlDeviceGetHandleByIndex_v2", "nvmlDeviceGetHandleByIndex"),
            [ctypes.c_uint, ctypes.POINTER(ctypes.c_void_p)],
        )
        self._device_name = _nvml_function(
            self._library,
            ("nvmlDeviceGetName",),
            [ctypes.c_void_p, ctypes.POINTER(ctypes.c_char), ctypes.c_uint],
        )
        self._device_uuid = _nvml_function(
            self._library,
            ("nvmlDeviceGetUUID",),
            [ctypes.c_void_p, ctypes.POINTER(ctypes.c_char), ctypes.c_uint],
        )
        self._device_temperature = _nvml_function(
            self._library,
            ("nvmlDeviceGetTemperature",),
            [ctypes.c_void_p, ctypes.c_uint, ctypes.POINTER(ctypes.c_uint)],
        )
        self._memory_info = _nvml_function(
            self._library,
            ("nvmlDeviceGetMemoryInfo",),
            [ctypes.c_void_p, ctypes.POINTER(_NvmlMemory)],
        )
        self._system_version = _nvml_function(
            self._library,
            ("nvmlSystemGetNVMLVersion",),
            [ctypes.POINTER(ctypes.c_char), ctypes.c_uint],
        )
        self._error_string = getattr(self._library, "nvmlErrorString", None)
        if self._error_string is not None:
            _configure(self._error_string, [ctypes.c_int], ctypes.c_char_p)

    def _check(self, result: int, operation: str) -> None:
        if result == 0:
            return
        raise DeviceMemoryError(
            f"{operation} failed: {self._error_detail(result)}"
        )

    def _error_detail(self, result: int) -> str:
        detail = f"NVML error {result}"
        if self._error_string is not None:
            raw = self._error_string(result)
            if raw:
                detail = raw.decode("utf-8", errors="replace")
        return detail

    def _read_string(self, operation: str, function: Any, size: int) -> str:
        return self._read_string_for_handle(
            self._handle, operation, function, size
        )

    def _read_string_for_handle(
        self,
        handle: ctypes.c_void_p,
        operation: str,
        function: Any,
        size: int,
    ) -> str:
        buffer = ctypes.create_string_buffer(size)
        self._check(function(handle, buffer, size), operation)
        return _decode_buffer(buffer, operation)

    def _handle_at(self, device_index: int) -> ctypes.c_void_p:
        handle = ctypes.c_void_p()
        self._check(
            self._device_by_index(device_index, ctypes.byref(handle)),
            "nvmlDeviceGetHandleByIndex_v2",
        )
        if handle.value is None:
            raise DeviceMemoryError("NVML returned a null device handle")
        return handle

    def _read_identity(self) -> dict[str, Any]:
        version_buffer = ctypes.create_string_buffer(80)
        self._check(
            self._system_version(version_buffer, len(version_buffer)),
            "nvmlSystemGetNVMLVersion",
        )
        memory = self._get_memory_info()
        return {
            "selector": self._selector,
            "index": self._device_index,
            "enumerated_device_count": self._device_count_value,
            "uuid": self._read_string("nvmlDeviceGetUUID", self._device_uuid, 96),
            "name": self._read_string("nvmlDeviceGetName", self._device_name, 96),
            "total_bytes": memory.total,
            "library": self._library_name,
            "nvml_version": _decode_buffer(
                version_buffer, "nvmlSystemGetNVMLVersion"
            ),
        }

    def _get_memory_info(self) -> _NvmlMemory:
        memory = _NvmlMemory()
        self._check(
            self._memory_info(self._handle, ctypes.byref(memory)),
            "nvmlDeviceGetMemoryInfo",
        )
        if memory.total <= 0 or memory.used > memory.total or memory.free > memory.total:
            raise DeviceMemoryError("NVML returned inconsistent device-memory values")
        return memory

    def read_bytes(self) -> int:
        if self._closed:
            raise DeviceMemoryError("NVML memory counter is closed")
        return self._get_memory_info().used

    def read_temperature_millicelsius(self) -> int:
        if self._closed:
            raise DeviceMemoryError("NVML memory counter is closed")
        temperature = ctypes.c_uint()
        self._check(
            self._device_temperature(self._handle, 0, ctypes.byref(temperature)),
            "nvmlDeviceGetTemperature",
        )
        if temperature.value <= 0 or temperature.value > 200:
            raise DeviceMemoryError(
                f"NVML returned implausible GPU temperature {temperature.value} C"
            )
        return temperature.value * 1000

    def receipt_identity(self) -> dict[str, Any]:
        return {"source": self.source, "path": None, "device": dict(self._identity)}

    def close(self) -> None:
        if not self._closed:
            self._closed = True
            result = self._shutdown()
            self._check(result, "nvmlShutdown")


def _drm_candidates() -> list[Path]:
    return sorted(
        Path(path).resolve()
        for path in glob.glob("/sys/class/drm/card*/device/mem_info_vram_used")
        if Path(path).is_file()
    )


def resolve_drm_counter(raw_path: str) -> DrmMemoryCounter:
    if raw_path != "auto":
        path = Path(raw_path).expanduser().resolve()
        if not path.is_file():
            raise DeviceMemoryError(f"DRM memory counter does not exist: {path}")
        return DrmMemoryCounter(path)
    candidates = _drm_candidates()
    if len(candidates) == 1:
        return DrmMemoryCounter(candidates[0])
    if not candidates:
        raise DeviceMemoryError("no DRM device-memory counter found")
    raise DeviceMemoryError(
        "multiple DRM memory counters found; select one with --memory-path: "
        + ", ".join(str(path) for path in candidates)
    )


def resolve_memory_counter(
    *,
    source: str,
    drm_path: str,
    nvml_device_index: int | None,
    nvml_device_uuid: str | None = None,
    nvml_library_loader: Callable[[str], Any] = ctypes.CDLL,
) -> DrmMemoryCounter | NvmlMemoryCounter:
    """Resolve one unambiguous whole-device memory counter."""

    if source == "drm":
        if nvml_device_index is not None or nvml_device_uuid is not None:
            raise DeviceMemoryError(
                "NVML device selectors cannot be combined with --memory-source drm"
            )
        return resolve_drm_counter(drm_path)
    if source == "nvml":
        if drm_path != "auto":
            raise DeviceMemoryError(
                "--memory-path cannot be combined with --memory-source nvml"
            )
        return NvmlMemoryCounter(
            nvml_device_index,
            device_uuid=nvml_device_uuid,
            library_loader=nvml_library_loader,
        )
    if source != "auto":
        raise DeviceMemoryError(f"unsupported device-memory source {source!r}")
    if (
        drm_path != "auto"
        or nvml_device_index is not None
        or nvml_device_uuid is not None
    ):
        raise DeviceMemoryError(
            "explicit memory selectors require --memory-source drm or nvml"
        )

    drm_candidates = _drm_candidates()
    nvml_counter: NvmlMemoryCounter | None = None
    nvml_error: DeviceMemoryError | None = None
    try:
        nvml_counter = NvmlMemoryCounter(
            None, library_loader=nvml_library_loader
        )
    except NvmlUnavailable as exc:
        nvml_error = exc
    except DeviceMemoryError as exc:
        if drm_candidates:
            raise DeviceMemoryError(
                "NVML is available but cannot be selected automatically while "
                f"DRM is also available: {exc}; select --memory-source explicitly"
            ) from exc
        raise

    if len(drm_candidates) == 1 and nvml_counter is None:
        return DrmMemoryCounter(drm_candidates[0])
    if not drm_candidates and nvml_counter is not None:
        return nvml_counter
    if nvml_counter is not None:
        nvml_counter.close()
    if len(drm_candidates) > 1:
        raise DeviceMemoryError(
            "multiple DRM memory counters found; select one with "
            "--memory-source drm --memory-path PATH"
        )
    if drm_candidates:
        raise DeviceMemoryError(
            "both DRM and NVML device-memory counters are available; select one "
            "with --memory-source"
        )
    detail = f"; NVML: {nvml_error}" if nvml_error is not None else ""
    raise DeviceMemoryError(f"no device-memory counter found{detail}")


class MemorySampler:
    """Background peak sampler that fails closed on any read error."""

    def __init__(
        self,
        counter: DrmMemoryCounter | NvmlMemoryCounter | Path | None,
        interval_ms: int,
    ) -> None:
        if interval_ms <= 0:
            raise DeviceMemoryError("memory sampling cadence must be positive")
        if isinstance(counter, Path):
            counter = DrmMemoryCounter(counter)
        self.counter = counter
        self.interval_secs = interval_ms / 1000.0
        self._stop = threading.Event()
        self._lock = threading.Lock()
        self._thread: threading.Thread | None = None
        self._baseline: int | None = None
        self._peak: int | None = None
        self._samples = 0
        self._error: Exception | None = None

    def _read(self) -> int:
        if self.counter is None:
            raise DeviceMemoryError("memory sampler is disabled")
        return self.counter.read_bytes()

    def start(self) -> None:
        if self.counter is None:
            return
        self._stop.clear()
        self.reset()
        self._thread = threading.Thread(
            target=self._run, name="benchmark-memory", daemon=True
        )
        self._thread.start()

    def _run(self) -> None:
        while not self._stop.wait(self.interval_secs):
            try:
                with self._lock:
                    self._raise_sampling_error_locked()
                    value = self._read()
                    self._peak = (
                        value if self._peak is None else max(self._peak, value)
                    )
                    self._samples += 1
            except Exception as exc:
                with self._lock:
                    if self._error is None:
                        self._error = exc
                self._stop.set()
                return

    def _raise_sampling_error(self) -> None:
        with self._lock:
            self._raise_sampling_error_locked()

    def _raise_sampling_error_locked(self) -> None:
        error = self._error
        if error is not None:
            raise DeviceMemoryError(
                f"device-memory sampling failed: {type(error).__name__}: {error}"
            ) from error

    def reset(self) -> None:
        if self.counter is None:
            return
        with self._lock:
            self._raise_sampling_error_locked()
            value = self._read()
            self._baseline = value
            self._peak = value
            self._samples = 1

    def snapshot(self) -> dict[str, int] | None:
        if self.counter is None:
            return None
        with self._lock:
            self._raise_sampling_error_locked()
            value = self._read()
            peak = max(self._peak if self._peak is not None else value, value)
            baseline = (
                self._baseline if self._baseline is not None else value
            )
            samples = self._samples + 1
        return {
            "baseline_bytes": baseline,
            "peak_bytes": peak,
            "peak_delta_bytes": max(0, peak - baseline),
            "samples": samples,
        }

    def receipt_identity(self) -> dict[str, Any]:
        if self.counter is None:
            return {"source": "unavailable", "path": None, "device": None}
        return self.counter.receipt_identity()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            if self._thread.is_alive():
                raise DeviceMemoryError("device-memory sampler thread did not stop")
        sampling_error: Exception | None = None
        try:
            self._raise_sampling_error()
        except Exception as exc:
            sampling_error = exc
        close_error: Exception | None = None
        if self.counter is not None:
            try:
                self.counter.close()
            except Exception as exc:
                close_error = exc
        if sampling_error is not None:
            raise sampling_error
        if close_error is not None:
            raise DeviceMemoryError(
                f"device-memory counter close failed: {close_error}"
            ) from close_error
