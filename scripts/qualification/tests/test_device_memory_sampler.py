from __future__ import annotations

import ctypes
import importlib.util
import tempfile
import threading
import time
import unittest
from pathlib import Path
from unittest import mock


ROOT = Path(__file__).resolve().parents[3]
SCRIPT = ROOT / "scripts" / "qualification" / "device_memory_sampler.py"
SPEC = importlib.util.spec_from_file_location("device_memory_sampler", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
memory = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(memory)


class FakeFunction:
    def __init__(self, callback):
        self.callback = callback
        self.argtypes = None
        self.restype = None

    def __call__(self, *args):
        return self.callback(*args)


class FakeNvml:
    _name = "fixture-libnvidia-ml.so.1"

    def __init__(
        self,
        *,
        device_count: int = 1,
        fail_after_reads: int | None = None,
        init_result: int = 0,
    ):
        self.device_count = device_count
        self.fail_after_reads = fail_after_reads
        self.memory_reads = 0
        self.shutdowns = 0
        self.nvmlInit_v2 = FakeFunction(lambda: init_result)
        self.nvmlShutdown = FakeFunction(self._shutdown)
        self.nvmlDeviceGetCount_v2 = FakeFunction(self._get_count)
        self.nvmlDeviceGetHandleByIndex_v2 = FakeFunction(self._get_handle)
        self.nvmlDeviceGetName = FakeFunction(
            lambda _handle, target, size: self._write_string(
                target, size, b"NVIDIA GeForce RTX 4090"
            )
        )
        self.nvmlDeviceGetUUID = FakeFunction(self._get_uuid)
        self.nvmlDeviceGetMemoryInfo = FakeFunction(self._get_memory)
        self.nvmlDeviceGetTemperature = FakeFunction(self._get_temperature)
        self.nvmlSystemGetNVMLVersion = FakeFunction(
            lambda target, size: self._write_string(target, size, b"13.580.65")
        )
        self.nvmlErrorString = FakeFunction(lambda code: f"fixture-{code}".encode())

    def _shutdown(self):
        self.shutdowns += 1
        return 0

    def _get_count(self, target):
        ctypes.cast(target, ctypes.POINTER(ctypes.c_uint)).contents.value = (
            self.device_count
        )
        return 0

    def _get_handle(self, index, target):
        if int(index) >= self.device_count:
            return 6
        ctypes.cast(target, ctypes.POINTER(ctypes.c_void_p)).contents.value = (
            0x4090 + int(index)
        )
        return 0

    def _get_uuid(self, handle, target, size):
        index = handle.value - 0x4090
        value = (
            b"GPU-01234567-89ab-cdef-0123-456789abcdef"
            if index == 0
            else b"GPU-11234567-89ab-cdef-0123-456789abcdef"
        )
        return self._write_string(target, size, value)

    @staticmethod
    def _write_string(target, size, value: bytes):
        if len(value) + 1 > int(size):
            return 7
        ctypes.memmove(target, value + b"\0", len(value) + 1)
        return 0

    def _get_memory(self, _handle, target):
        self.memory_reads += 1
        if (
            self.fail_after_reads is not None
            and self.memory_reads > self.fail_after_reads
        ):
            return 999
        info = ctypes.cast(
            target, ctypes.POINTER(memory._NvmlMemory)
        ).contents
        info.total = 24 * 1024**3
        info.free = 20 * 1024**3 - self.memory_reads
        info.used = 4 * 1024**3 + self.memory_reads
        return 0

    @staticmethod
    def _get_temperature(_handle, _sensor, target):
        ctypes.cast(target, ctypes.POINTER(ctypes.c_uint)).contents.value = 63
        return 0


class DeviceMemorySamplerTests(unittest.TestCase):
    def test_nvml_counter_records_stable_device_identity(self) -> None:
        library = FakeNvml()
        counter = memory.NvmlMemoryCounter(
            None, library_loader=lambda _name: library
        )
        identity = counter.receipt_identity()
        used = counter.read_bytes()
        temperature = counter.read_temperature_millicelsius()
        counter.close()

        self.assertEqual(identity["source"], "nvml_used")
        self.assertIsNone(identity["path"])
        self.assertEqual(identity["device"]["selector"], "auto_single_device")
        self.assertEqual(identity["device"]["index"], 0)
        self.assertEqual(identity["device"]["enumerated_device_count"], 1)
        self.assertEqual(identity["device"]["name"], "NVIDIA GeForce RTX 4090")
        self.assertTrue(identity["device"]["uuid"].startswith("GPU-"))
        self.assertEqual(identity["device"]["total_bytes"], 24 * 1024**3)
        self.assertGreater(used, 4 * 1024**3)
        self.assertEqual(temperature, 63_000)
        self.assertEqual(library.shutdowns, 1)

    def test_nvml_multi_gpu_auto_selection_fails_and_shuts_down(self) -> None:
        library = FakeNvml(device_count=2)
        with self.assertRaisesRegex(
            memory.DeviceMemoryError, "select one with --memory-device-index"
        ):
            memory.NvmlMemoryCounter(None, library_loader=lambda _name: library)
        self.assertEqual(library.shutdowns, 1)

    def test_nvml_temperature_error_and_closed_counter_fail_closed(self) -> None:
        library = FakeNvml()
        library.nvmlDeviceGetTemperature = FakeFunction(
            lambda _handle, _sensor, _target: 999
        )
        counter = memory.NvmlMemoryCounter(
            None, library_loader=lambda _name: library
        )
        with self.assertRaisesRegex(
            memory.DeviceMemoryError, "fixture-999"
        ):
            counter.read_temperature_millicelsius()
        counter.close()
        with self.assertRaisesRegex(
            memory.DeviceMemoryError, "closed"
        ):
            counter.read_temperature_millicelsius()

    def test_explicit_nvml_index_is_bound_in_identity(self) -> None:
        library = FakeNvml(device_count=2)
        counter = memory.resolve_memory_counter(
            source="nvml",
            drm_path="auto",
            nvml_device_index=1,
            nvml_library_loader=lambda _name: library,
        )
        self.assertEqual(counter.receipt_identity()["device"]["selector"], "explicit_index")
        self.assertEqual(counter.receipt_identity()["device"]["index"], 1)
        counter.close()

    def test_explicit_nvml_uuid_resolves_independently_of_index(self) -> None:
        library = FakeNvml(device_count=2)
        counter = memory.resolve_memory_counter(
            source="nvml",
            drm_path="auto",
            nvml_device_index=None,
            nvml_device_uuid="GPU-11234567-89ab-cdef-0123-456789abcdef",
            nvml_library_loader=lambda _name: library,
        )
        identity = counter.receipt_identity()["device"]
        self.assertEqual(identity["selector"], "explicit_uuid")
        self.assertEqual(identity["index"], 1)
        self.assertEqual(
            identity["uuid"], "GPU-11234567-89ab-cdef-0123-456789abcdef"
        )
        counter.close()

    def test_auto_rejects_mixed_drm_and_nvml_sources(self) -> None:
        library = FakeNvml()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "mem_info_vram_used"
            path.write_text("1")
            with mock.patch.object(memory, "_drm_candidates", return_value=[path]):
                with self.assertRaisesRegex(
                    memory.DeviceMemoryError, "both DRM and NVML"
                ):
                    memory.resolve_memory_counter(
                        source="auto",
                        drm_path="auto",
                        nvml_device_index=None,
                        nvml_library_loader=lambda _name: library,
                    )
        self.assertEqual(library.shutdowns, 1)

    def test_auto_does_not_hide_ambiguous_nvml_behind_one_drm_device(self) -> None:
        library = FakeNvml(device_count=2)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "mem_info_vram_used"
            path.write_text("1")
            with mock.patch.object(memory, "_drm_candidates", return_value=[path]):
                with self.assertRaisesRegex(
                    memory.DeviceMemoryError,
                    "NVML is available but cannot be selected automatically",
                ):
                    memory.resolve_memory_counter(
                        source="auto",
                        drm_path="auto",
                        nvml_device_index=None,
                        nvml_library_loader=lambda _name: library,
                    )
        self.assertEqual(library.shutdowns, 1)

    def test_auto_uses_drm_when_nvml_cannot_initialize(self) -> None:
        library = FakeNvml(init_result=9)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "mem_info_vram_used"
            path.write_text("123")
            with mock.patch.object(memory, "_drm_candidates", return_value=[path]):
                counter = memory.resolve_memory_counter(
                    source="auto",
                    drm_path="auto",
                    nvml_device_index=None,
                    nvml_library_loader=lambda _name: library,
                )
            self.assertEqual(counter.read_bytes(), 123)
            counter.close()
        self.assertEqual(library.shutdowns, 0)

    def test_background_sampling_error_fails_closed(self) -> None:
        library = FakeNvml(fail_after_reads=3)
        counter = memory.NvmlMemoryCounter(
            0, library_loader=lambda _name: library
        )
        sampler = memory.MemorySampler(counter, 1)
        sampler.start()
        time.sleep(0.02)
        with self.assertRaisesRegex(memory.DeviceMemoryError, "sampling failed"):
            sampler.snapshot()
        with self.assertRaisesRegex(memory.DeviceMemoryError, "sampling failed"):
            sampler.stop()
        self.assertEqual(library.shutdowns, 1)

    def test_run_boundary_cannot_inherit_an_inflight_old_sample(self) -> None:
        class BlockingCounter:
            def __init__(self):
                self.calls = 0
                self.background_entered = threading.Event()
                self.release_background = threading.Event()

            def read_bytes(self):
                self.calls += 1
                if self.calls == 1:
                    return 100
                if self.calls == 2:
                    self.background_entered.set()
                    self.release_background.wait(timeout=1.0)
                    return 999
                return 200

            def receipt_identity(self):
                return {"source": "fixture", "path": None, "device": None}

            def close(self):
                return None

        counter = BlockingCounter()
        sampler = memory.MemorySampler(counter, 1)
        sampler.start()
        self.assertTrue(counter.background_entered.wait(timeout=1.0))
        reset = threading.Thread(target=sampler.reset)
        reset.start()
        time.sleep(0.01)
        self.assertTrue(reset.is_alive())
        counter.release_background.set()
        reset.join(timeout=1.0)
        self.assertFalse(reset.is_alive())
        snapshot = sampler.snapshot()
        sampler.stop()
        self.assertEqual(snapshot["baseline_bytes"], 200)
        self.assertEqual(snapshot["peak_bytes"], 200)


if __name__ == "__main__":
    unittest.main()
