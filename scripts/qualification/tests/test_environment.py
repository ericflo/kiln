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


if __name__ == "__main__":
    unittest.main()
