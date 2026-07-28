from __future__ import annotations

import ctypes
import importlib.util
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock


QUALIFICATION_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(QUALIFICATION_DIR))
SPEC = importlib.util.spec_from_file_location(
    "qualification_wsl_platform",
    QUALIFICATION_DIR / "wsl_platform.py",
)
assert SPEC is not None and SPEC.loader is not None
wsl_platform = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = wsl_platform
SPEC.loader.exec_module(wsl_platform)


WSL_VERSION = """WSL version: 2.5.9.0
Kernel version: 6.6.87.2-1
WSLg version: 1.0.66
MSRDC version: 1.2.6074
Direct3D version: 1.611.1-81528511
DXCore version: 10.0.26100.1-240331-1435.ge-release
Windows version: 10.0.26200.8655
"""


class FakeFunction:
    def __init__(self, name: str, callback: object) -> None:
        self.__name__ = name
        self.callback = callback

    def __call__(self, *args: object) -> object:
        return self.callback(*args)  # type: ignore[operator]


class WslPlatformTests(unittest.TestCase):
    def test_wsl2_detection_requires_kernel_and_interop(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            interop = Path(tmp) / "WSLInterop"
            interop.write_text("enabled\ninterpreter /init\n")
            with mock.patch.dict(os.environ, {}, clear=True):
                self.assertTrue(
                    wsl_platform.is_wsl2(
                        kernel_release="6.6.87.2-microsoft-standard-WSL2",
                        interop_path=interop,
                    )
                )
                self.assertFalse(
                    wsl_platform.is_wsl2(
                        kernel_release="6.6.87-generic",
                        interop_path=interop,
                    )
                )
                interop.write_text("disabled\n")
                self.assertFalse(
                    wsl_platform.is_wsl2(
                        kernel_release="6.6.87.2-microsoft-standard-WSL2",
                        interop_path=interop,
                    )
                )

    def test_windows_output_and_wsl_versions_are_strict(self) -> None:
        encoded = ("\ufeff" + WSL_VERSION).encode("utf-16-le")
        parsed = wsl_platform.parse_wsl_version(
            wsl_platform.decode_windows_output(encoded)
        )
        self.assertEqual(parsed["wsl_version"], "2.5.9.0")
        self.assertEqual(parsed["kernel_version"], "6.6.87.2-1")
        self.assertEqual(parsed["windows_version"], "10.0.26200.8655")
        with self.assertRaisesRegex(
            wsl_platform.PlatformProbeError,
            "omitted",
        ):
            wsl_platform.parse_wsl_version("WSL version: 2.5.9.0\n")

    def test_windows_nvidia_driver_mapping_is_exact(self) -> None:
        self.assertEqual(
            wsl_platform.windows_nvidia_driver_version("32.0.15.9636"),
            "596.36",
        )
        self.assertEqual(
            wsl_platform.windows_nvidia_driver_version("32.0.15.8095"),
            "580.95",
        )
        with self.assertRaisesRegex(
            wsl_platform.PlatformProbeError,
            "invalid",
        ):
            wsl_platform.windows_nvidia_driver_version("596.36")

    def test_windows_formatted_thermal_zone_is_strict_and_converted(self) -> None:
        row = {
            "Name": "\\_TZ.THRM",
            "Temperature": 345,
            "HighPrecisionTemperature": 3452,
            "PercentPassiveLimit": 100,
            "ThrottleReasons": 0,
        }
        parsed = wsl_platform.parse_windows_thermal_zones(json.dumps(row))
        self.assertEqual(parsed[0]["temperature_millicelsius"], 72_050)
        row["HighPrecisionTemperature"] = 3600
        with self.assertRaisesRegex(
            wsl_platform.PlatformProbeError,
            "disagree",
        ):
            wsl_platform.parse_windows_thermal_zones(json.dumps(row))
        row["HighPrecisionTemperature"] = 3452
        row["ThrottleReasons"] = -1
        with self.assertRaisesRegex(
            wsl_platform.PlatformProbeError,
            "invalid counters",
        ):
            wsl_platform.parse_windows_thermal_zones(json.dumps(row))

    def test_hwmon_temperatures_are_typed_and_bounded(self) -> None:
        with tempfile.TemporaryDirectory() as raw_directory:
            root = Path(raw_directory)
            hwmon = root / "hwmon0"
            hwmon.mkdir()
            (hwmon / "name").write_text("coretemp\n")
            (hwmon / "temp1_label").write_text("Package id 0\n")
            temperature = hwmon / "temp1_input"
            temperature.write_text("64000\n")
            self.assertEqual(
                wsl_platform._hwmon_temperatures(root),
                [
                    {
                        "hwmon_name": "coretemp",
                        "label": "Package id 0",
                        "input_path": str(temperature.resolve()),
                        "temperature_millicelsius": 64_000,
                    }
                ],
            )
            temperature.write_text("300000\n")
            with self.assertRaisesRegex(
                wsl_platform.PlatformProbeError,
                "implausible",
            ):
                wsl_platform._hwmon_temperatures(root)

    def test_nvml_probe_reads_temperature_and_shuts_down(self) -> None:
        shutdowns: list[bool] = []

        def text(value: bytes) -> object:
            def fill(*args: object) -> int:
                buffer = args[-2]
                buffer.value = value  # type: ignore[attr-defined]
                return 0

            return fill

        def set_uint(value: int) -> object:
            def fill(target: object) -> int:
                ctypes.cast(target, ctypes.POINTER(ctypes.c_uint))[0] = value
                return 0

            return fill

        def memory(_handle: object, target: object) -> int:
            value = ctypes.cast(
                target,
                ctypes.POINTER(wsl_platform.NvmlMemory),
            )[0]
            value.total = 16_000
            value.free = 12_000
            value.used = 4_000
            return 0

        library = SimpleNamespace()
        library.nvmlInit = library.nvmlInit_v2 = FakeFunction(
            "nvmlInit_v2", lambda: 0
        )
        library.nvmlShutdown = FakeFunction(
            "nvmlShutdown", lambda: shutdowns.append(True) or 0
        )
        library.nvmlSystemGetNVMLVersion = FakeFunction(
            "nvmlSystemGetNVMLVersion", text(b"13.1")
        )
        library.nvmlSystemGetDriverVersion = FakeFunction(
            "nvmlSystemGetDriverVersion", text(b"596.36")
        )
        library.nvmlDeviceGetCount = library.nvmlDeviceGetCount_v2 = (
            FakeFunction("nvmlDeviceGetCount_v2", set_uint(1))
        )
        library.nvmlDeviceGetHandleByIndex = (
            library.nvmlDeviceGetHandleByIndex_v2
        ) = FakeFunction(
            "nvmlDeviceGetHandleByIndex_v2",
            lambda _index, target: (
                ctypes.cast(target, ctypes.POINTER(ctypes.c_void_p)).__setitem__(
                    0, 1
                )
                or 0
            ),
        )
        library.nvmlDeviceGetUUID = FakeFunction(
            "nvmlDeviceGetUUID", text(b"GPU-1234")
        )
        library.nvmlDeviceGetName = FakeFunction(
            "nvmlDeviceGetName", text(b"NVIDIA Test GPU")
        )
        library.nvmlDeviceGetMemoryInfo = FakeFunction(
            "nvmlDeviceGetMemoryInfo", memory
        )
        library.nvmlDeviceGetTemperature = FakeFunction(
            "nvmlDeviceGetTemperature",
            lambda _handle, sensor, target: (
                self.assertEqual(sensor, 0)
                or ctypes.cast(
                    target, ctypes.POINTER(ctypes.c_uint)
                ).__setitem__(0, 67)
                or 0
            ),
        )

        with mock.patch.object(
            wsl_platform.ctypes,
            "CDLL",
            return_value=library,
        ):
            value = wsl_platform._nvml_probe(0, {})
        self.assertEqual(value["temperature_c"], 67)
        self.assertEqual(shutdowns, [True])

    def test_containment_binding_is_fail_closed(self) -> None:
        platform_value = {
            "kind": "wsl2",
            "capabilities": {
                key: "unavailable" for key in wsl_platform.CAPABILITY_KEYS
            },
            "details": {"wsl_identity": "test"},
        }
        results = [
            {
                "id": "wsl2-workload-containment",
                "required": True,
                "status": "failed",
                "duration_seconds": 0.0,
                "metrics": [],
                "details": "missing",
            }
        ]
        wsl_platform.bind_containment(platform_value, results, "untrusted")
        self.assertEqual(
            platform_value["capabilities"]["network_containment"],
            "unavailable",
        )
        self.assertEqual(results[0]["status"], "failed")
        mechanism = next(iter(wsl_platform.WSL_CONTAINMENT_MECHANISMS))
        wsl_platform.bind_containment(platform_value, results, mechanism)
        self.assertEqual(
            platform_value["capabilities"]["network_containment"],
            "available",
        )
        self.assertEqual(
            platform_value["capabilities"]["process_containment"],
            "available",
        )
        self.assertEqual(results[0]["status"], "passed")


if __name__ == "__main__":
    unittest.main()
