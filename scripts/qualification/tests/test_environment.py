from __future__ import annotations

import importlib.util
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock


QUALIFICATION_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(QUALIFICATION_DIR))
SPEC = importlib.util.spec_from_file_location("qualification_environment", QUALIFICATION_DIR / "environment.py")
assert SPEC is not None and SPEC.loader is not None
environment = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = environment
SPEC.loader.exec_module(environment)


ROCMINFO = """
Runtime Version:         1.18
*******
Agent 1
*******
  Name:                    gfx1151
  Marketing Name:          AMD Radeon 8060S Graphics
  Device Type:             GPU
  Chip ID:                 5510(0x1586)
  Compute Unit:            40
  Memory Properties:       APU
  Wavefront Size:          32(0x20)
*******
"""

VULKANINFO = """
Vulkan Instance Version: 1.4.350
GPU0:
    apiVersion         = 1.4.348
    driverVersion      = 26.1.3
    vendorID           = 0x1002
    deviceID           = 0x1586
    deviceType         = PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU
    deviceName         = AMD Radeon 8060S Graphics (RADV STRIX_HALO)
    driverID           = DRIVER_ID_MESA_RADV
    driverName         = radv
    driverInfo         = Mesa 26.1.3-arch1.2
"""

NVIDIA_SMI = """0, NVIDIA GeForce RTX 4090 Laptop GPU, GPU-1234, 00000000:01:00.0, 8.9, 16376, 15001, 580.95
1, NVIDIA GeForce RTX 4090, GPU-5678, 00000000:02:00.0, 8.9, 24564, 24000, 580.95
"""

METAL_SYSTEM_PROFILER = """
{
  "SPDisplaysDataType": [
    {
      "_name": "Apple M1",
      "sppci_model": "Apple M1",
      "sppci_cores": "8",
      "spdisplays_mtlgpufamilysupport": "spdisplays_metal3"
    }
  ]
}
"""

SW_VERS = """ProductName:\t\tmacOS
ProductVersion:\t\t15.5
BuildVersion:\t\t24F74
"""


class EnvironmentTests(unittest.TestCase):
    def test_parse_rocm_gpu_agent(self) -> None:
        agent = environment.parse_rocm_agent(ROCMINFO)
        self.assertIsNotNone(agent)
        assert agent is not None
        self.assertEqual(agent["architecture"], "gfx1151")
        self.assertEqual(agent["device_id"], 0x1586)
        self.assertEqual(agent["compute_units"], "40")
        self.assertEqual(agent["wavefront_size"], "32")
        self.assertTrue(agent["unified_memory"])

    def test_parse_vulkan_physical_device(self) -> None:
        device = environment.parse_vulkan_summary(VULKANINFO)
        self.assertIsNotNone(device)
        assert device is not None
        self.assertEqual(device["vendor_id"], 0x1002)
        self.assertEqual(device["device_id"], 0x1586)
        self.assertEqual(device["architecture"], "strix_halo")
        self.assertTrue(device["integrated"])
        self.assertEqual(device["logical_index"], 0)

    def test_parse_nvidia_devices_preserves_selection_and_memory_identity(self) -> None:
        devices = environment.parse_nvidia_smi_devices(NVIDIA_SMI)
        self.assertEqual(len(devices), 2)
        laptop, desktop = devices
        self.assertEqual(laptop["logical_index"], 0)
        self.assertEqual(laptop["architecture"], "sm_89")
        self.assertEqual(laptop["compute_capability"], "8.9")
        self.assertEqual(laptop["memory_bytes"], 16376 * 1024**2)
        self.assertEqual(laptop["memory_available_bytes"], 15001 * 1024**2)
        self.assertEqual(desktop["device_uuid"], "GPU-5678")

    def test_parse_nvidia_devices_rejects_ambiguous_or_invalid_rows(self) -> None:
        with self.assertRaisesRegex(ValueError, "duplicate"):
            environment.parse_nvidia_smi_devices(NVIDIA_SMI + NVIDIA_SMI.splitlines()[0] + "\n")
        with self.assertRaisesRegex(ValueError, "invalid index or memory"):
            environment.parse_nvidia_smi_devices(
                "0, GPU, GPU-1, 0000:01:00.0, 8.9, 16000, 17000, 580.95\n"
            )

    def test_parse_metal_device_and_macos_version(self) -> None:
        device = environment.parse_metal_device(
            METAL_SYSTEM_PROFILER, str(16 * 1024**3)
        )
        self.assertIsNotNone(device)
        assert device is not None
        self.assertEqual(device["name"], "Apple M1")
        self.assertEqual(device["architecture"], "apple_m1")
        self.assertEqual(device["compute_units"], 8)
        self.assertEqual(device["memory_bytes"], 16 * 1024**3)
        self.assertEqual(
            environment.parse_sw_vers(SW_VERS),
            {
                "product_name": "macOS",
                "product_version": "15.5",
                "build_version": "24F74",
            },
        )

    def test_device_class_expectations_are_fail_closed(self) -> None:
        laptop = environment.parse_nvidia_smi_devices(NVIDIA_SMI)[0]
        passed = environment.device_expectation_result(
            laptop,
            expected_name_regex=r"^NVIDIA GeForce RTX 4090 Laptop GPU$",
            expected_compute_units=None,
            minimum_memory_mib=15000,
            maximum_memory_mib=17000,
        )
        self.assertIsNotNone(passed)
        assert passed is not None
        self.assertEqual(passed["status"], "passed")
        failed = environment.device_expectation_result(
            laptop,
            expected_name_regex=r"^NVIDIA GeForce RTX 4090$",
            expected_compute_units=None,
            minimum_memory_mib=23000,
            maximum_memory_mib=25000,
        )
        self.assertIsNotNone(failed)
        assert failed is not None
        self.assertEqual(failed["status"], "failed")
        self.assertIn("does not fully match", failed["details"])
        self.assertIn("below 23000 MiB", failed["details"])
        wrong_core_count = environment.device_expectation_result(
            {"name": "Apple M1", "compute_units": 7, "memory_bytes": 8 * 1024**3},
            expected_name_regex=r"^Apple M1$",
            expected_compute_units=8,
            minimum_memory_mib=None,
            maximum_memory_mib=None,
        )
        self.assertIsNotNone(wrong_core_count)
        assert wrong_core_count is not None
        self.assertEqual(wrong_core_count["status"], "failed")
        self.assertIn("does not equal 8", wrong_core_count["details"])

    def test_cuda_collector_requires_selected_device_and_toolkit(self) -> None:
        outputs = {
            "rustc-version": "rustc 1.96.0\n",
            "cargo-version": "cargo 1.96.0\n",
            "cuda-device-probe": NVIDIA_SMI,
            "nvcc-version": "Cuda compilation tools, release 12.8, V12.8.93\n",
        }

        def probe(probe_id, argv, raw, **kwargs):
            return (
                {
                    "id": probe_id,
                    "required": True,
                    "status": "passed",
                    "duration_seconds": 0.0,
                    "metrics": [],
                    "details": None,
                },
                outputs[probe_id],
            )

        with mock.patch.object(
            environment, "executable", side_effect=lambda name, *paths: name
        ), mock.patch.object(environment, "run_probe", side_effect=probe):
            device, runtime, compiler, results = environment.collect_backend(
                "cuda", {}, device_index=1
            )
        self.assertEqual(device["name"], "NVIDIA GeForce RTX 4090")
        self.assertEqual(device["logical_index"], 1)
        self.assertEqual(device["memory_available_bytes"], 24000 * 1024**2)
        self.assertEqual(runtime["cuda_driver"], "580.95")
        self.assertEqual(compiler["nvcc"], "release 12.8, V12.8.93")
        self.assertEqual(
            next(item for item in results if item["id"] == "cuda-selected-device")["status"],
            "passed",
        )

    def test_metal_collector_requires_gpu_unified_memory_and_toolchain(self) -> None:
        outputs = {
            "rustc-version": "rustc 1.96.0\n",
            "cargo-version": "cargo 1.96.0\n",
            "metal-device-probe": METAL_SYSTEM_PROFILER,
            "unified-memory-probe": str(8 * 1024**3),
            "macos-version": SW_VERS,
            "metal-compiler-path": "/Applications/Xcode.app/usr/bin/metal\n",
            "macos-sdk-version": "15.5\n",
            "apple-clang-version": "Apple clang version 17.0.0\n",
        }

        def probe(probe_id, argv, raw, **kwargs):
            return (
                {
                    "id": probe_id,
                    "required": True,
                    "status": "passed",
                    "duration_seconds": 0.0,
                    "metrics": [],
                    "details": None,
                },
                outputs[probe_id],
            )

        with mock.patch.object(
            environment, "executable", side_effect=lambda name, *paths: name
        ), mock.patch.object(environment, "run_probe", side_effect=probe):
            device, runtime, compiler, results = environment.collect_backend("metal", {})
        self.assertTrue(device["unified_memory"])
        self.assertEqual(device["memory_bytes"], 8 * 1024**3)
        self.assertEqual(runtime["macos_build"], "24F74")
        self.assertIn("metal", compiler)
        self.assertEqual(
            next(item for item in results if item["id"] == "metal-selected-device")["status"],
            "passed",
        )

    def test_find_drm_device_matches_vendor_and_product(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            amd = root / "card1" / "device"
            amd.mkdir(parents=True)
            (amd / "vendor").write_text("0x1002\n")
            (amd / "device").write_text("0x1586\n")
            (amd / "mem_info_vram_total").write_text("103079215104\n")
            other = root / "card2" / "device"
            other.mkdir(parents=True)
            (other / "vendor").write_text("0x10de\n")
            (other / "device").write_text("0x1234\n")
            self.assertEqual(environment.find_drm_device(0x1002, 0x1586, root), amd)

    def test_sensitive_environment_values_are_hashed(self) -> None:
        with mock.patch.dict(
            os.environ,
            {
                "KILN_SERVER_MAX_DECODE_BATCH": "8",
                "KILN_OAUTH_JSON": '{"refresh_token":"must-not-appear"}',
                "KILN_TRAINING_WEBHOOK_URL": "https://secret.example/token",
                "CUDA_VISIBLE_DEVICES": "0",
                "UNRELATED": "ignored",
            },
            clear=True,
        ):
            captured = environment.captured_environment()
        self.assertEqual(captured["KILN_SERVER_MAX_DECODE_BATCH"], {"value": "8", "redacted": False})
        self.assertTrue(captured["KILN_OAUTH_JSON"]["redacted"])
        self.assertNotIn("must-not-appear", captured["KILN_OAUTH_JSON"]["value"])
        self.assertTrue(captured["KILN_TRAINING_WEBHOOK_URL"]["redacted"])
        self.assertRegex(captured["KILN_TRAINING_WEBHOOK_URL"]["value"], r"^sha256:[0-9a-f]{64}$")
        self.assertNotIn("UNRELATED", captured)
        self.assertEqual(
            captured["CUDA_VISIBLE_DEVICES"], {"value": "0", "redacted": False}
        )


if __name__ == "__main__":
    unittest.main()
