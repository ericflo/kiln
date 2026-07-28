from __future__ import annotations

import importlib.util
import errno
import json
import sys
import unittest
from pathlib import Path
from unittest import mock


QUALIFICATION_DIR = Path(__file__).resolve().parents[1]
ROOT = QUALIFICATION_DIR.parents[1]
sys.path.insert(0, str(QUALIFICATION_DIR))
SPEC = importlib.util.spec_from_file_location(
    "qualification_macos_platform",
    QUALIFICATION_DIR / "macos_platform.py",
)
assert SPEC is not None and SPEC.loader is not None
macos = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = macos
SPEC.loader.exec_module(macos)


HARDWARE_PROFILE = json.dumps(
    {
        "SPHardwareDataType": [
            {
                "machine_name": "MacBook Air",
                "machine_model": "MacBookAir10,1",
                "chip_type": "Apple M1",
                "physical_memory": "16 GB",
                "serial_number": "must-not-be-retained",
                "platform_UUID": "must-not-be-retained",
            }
        ]
    }
)


class MacosPlatformTests(unittest.TestCase):
    def test_hardware_profile_selects_nonsecret_crosscheck_fields(self) -> None:
        parsed = macos.parse_hardware_profile(HARDWARE_PROFILE)
        self.assertEqual(
            parsed,
            {
                "machine_name": "MacBook Air",
                "machine_model": "MacBookAir10,1",
                "chip_type": "Apple M1",
                "physical_memory": "16 GB",
            },
        )
        self.assertNotIn("serial_number", parsed)
        self.assertNotIn("platform_UUID", parsed)
        self.assertEqual(macos.parse_memory_size("16 GB"), 16 * 1024**3)

    def test_memory_observation_parsers_are_fail_closed(self) -> None:
        vm = macos.parse_vm_stat(
            """Mach Virtual Memory Statistics: (page size of 16384 bytes)
Pages free: 100.
Pages active: 200.
Pages inactive: 300.
Pages speculative: 40.
Pages wired down: 50.
Pages occupied by compressor: 60.
Pageins: 70.
Pageouts: 80.
Swapins: 90.
Swapouts: 100.
"""
        )
        self.assertEqual(vm["page_size_bytes"], 16384)
        self.assertEqual(vm["pages_occupied_by_compressor"], 60)
        pressure = macos.parse_memory_pressure(
            "The system has 17179869184 "
            "(1048576 pages with a page size of 16384).\n"
            "System-wide memory free percentage: 70%\n"
        )
        self.assertEqual(pressure["total_bytes"], 16 * 1024**3)
        self.assertEqual(pressure["free_percent"], 70)
        swap = macos.parse_swapusage(
            "total = 4096.00M  used = 1024.00M  free = 3072.00M  (encrypted)"
        )
        self.assertEqual(swap["used_bytes"], 1024 * 1024**2)
        self.assertTrue(swap["encrypted"])
        with self.assertRaises(macos.PlatformProbeError):
            macos.parse_memory_pressure("System-wide memory free percentage: 70%")

    def test_metal_runtime_requires_unified_memory_and_closed_fields(self) -> None:
        value = {
            "name": "Apple M1",
            "has_unified_memory": True,
            "max_buffer_length_bytes": 8 * 1024**3,
            "recommended_max_working_set_bytes": 11 * 1024**3,
            "current_allocated_bytes": 4096,
        }
        self.assertEqual(macos.parse_metal_runtime(json.dumps(value)), value)
        value["has_unified_memory"] = False
        with self.assertRaisesRegex(macos.PlatformProbeError, "not unified-memory"):
            macos.parse_metal_runtime(json.dumps(value))

    def test_containment_binding_accepts_only_macos_sandbox(self) -> None:
        platform_value = {
            "capabilities": {
                "network_containment": "unavailable",
                "process_containment": "unavailable",
            },
            "details": {},
        }
        results = [
            {
                "id": "macos-workload-containment",
                "status": "failed",
                "details": None,
            }
        ]
        macos.bind_containment(
            platform_value,
            results,
            "macos-sandbox-loopback-only-v1",
        )
        self.assertEqual(
            platform_value["capabilities"]["network_containment"],
            "available",
        )
        self.assertEqual(
            platform_value["capabilities"]["process_containment"],
            "available",
        )
        self.assertEqual(results[0]["status"], "passed")
        macos.bind_containment(platform_value, results, "unknown")
        self.assertEqual(
            platform_value["capabilities"]["network_containment"],
            "unavailable",
        )
        self.assertEqual(results[0]["status"], "failed")

    def test_contained_case_requires_loopback_denial_and_private_session(self) -> None:
        listener = mock.Mock()
        listener.getsockname.return_value = ("127.0.0.1", 1234)
        accepted = mock.Mock()
        listener.accept.return_value = (accepted, ("127.0.0.1", 2345))
        client = mock.Mock()
        external = mock.Mock()
        external.connect_ex.return_value = errno.EPERM
        with mock.patch.object(
            macos.socket,
            "socket",
            side_effect=(listener, client, external),
        ), mock.patch.object(
            macos.os, "getpid", return_value=123
        ), mock.patch.object(
            macos.os, "getpgrp", return_value=123
        ), mock.patch.object(
            macos.os, "getsid", return_value=123
        ):
            observed = macos.verify_contained_case(
                "macos-sandbox-loopback-only-v1"
            )
        self.assertEqual(observed["external_connect_errno"], errno.EPERM)
        client.connect.assert_called_once_with(("127.0.0.1", 1234))
        with self.assertRaisesRegex(
            macos.PlatformProbeError,
            "unrecognized macOS containment mechanism",
        ):
            macos.verify_contained_case("unknown")

    def test_collector_records_temperatures_as_explicitly_unavailable(self) -> None:
        selected = {
            "name": "Apple M1",
            "memory_bytes": 16 * 1024**3,
            "compute_units": 8,
        }
        hardware = {
            "machine_name": "MacBook Air",
            "machine_model": "MacBookAir10,1",
            "chip_type": "Apple M1",
            "cpu_brand": "Apple M1",
            "gpu_core_count": 8,
            "physical_memory_bytes": 16 * 1024**3,
            "kernel_build": "25G72",
        }
        runtime = {
            "name": "Apple M1",
            "has_unified_memory": True,
            "max_buffer_length_bytes": 8 * 1024**3,
            "recommended_max_working_set_bytes": 11 * 1024**3,
            "current_allocated_bytes": 0,
        }
        toolchain = {
            "xcode_version": "Xcode 26.6\nBuild version 17F113",
            "sdk_version": "26.5",
            "sdk_build": "25F70",
        }
        filesystem = {
            "root": str(ROOT),
            "source": "/dev/disk3s5",
            "fstype": "apfs",
            "mount_point": "/System/Volumes/Data",
            "case_sensitive": False,
        }
        memory = {
            "total_bytes": 16 * 1024**3,
            "free_percent": 70,
            "swap": {
                "total_bytes": 0,
                "used_bytes": 0,
                "encrypted": True,
            },
            "vm_stat": {
                "page_size_bytes": 16384,
                "pages_free": 1,
                "pages_active": 2,
                "pages_inactive": 3,
                "pages_wired_down": 4,
                "pages_occupied_by_compressor": 5,
                "pageins": 6,
                "pageouts": 7,
                "swapins": 8,
                "swapouts": 9,
            },
        }
        with mock.patch.object(
            macos, "sys_platform", return_value="darwin"
        ), mock.patch.object(
            macos, "_hardware_probe", return_value=hardware
        ), mock.patch.object(
            macos, "_metal_runtime_probe", return_value=runtime
        ), mock.patch.object(
            macos, "_toolchain_probe", return_value=toolchain
        ), mock.patch.object(
            macos,
            "_metal_compiler_probe",
            return_value={"air_bytes": 1, "metallib_bytes": 2},
        ), mock.patch.object(
            macos, "_filesystem_probe", return_value=filesystem
        ), mock.patch.object(
            macos, "_memory_probe", return_value=memory
        ), mock.patch.object(
            macos,
            "_thermal_probe",
            return_value={
                "thermal_warning": "not_recorded",
                "performance_warning": "not_recorded",
                "cpu_power_status": "not_recorded",
            },
        ):
            platform_value, results, unsupported = macos.collect(selected, {})
        assert platform_value is not None
        self.assertEqual(platform_value["kind"], "macos")
        self.assertEqual(
            set(platform_value["capabilities"]),
            macos.CAPABILITY_KEYS,
        )
        self.assertEqual(
            platform_value["capabilities"]["host_temperature"],
            "unavailable",
        )
        self.assertIsNone(platform_value["observations"]["gpu_temperature"])
        self.assertEqual(len(unsupported), 2)
        self.assertEqual(
            next(item for item in results if item["id"] == "macos-host-temperature")[
                "status"
            ],
            "skipped",
        )

    def test_receipt_schema_has_closed_macos_platform_contract(self) -> None:
        schema = json.loads(
            (ROOT / "qualification/schema/receipt-v1.schema.json").read_text()
        )
        alternatives = schema["properties"]["environment"]["properties"]["platform"][
            "oneOf"
        ]
        macos_schema = next(
            value
            for value in alternatives
            if value["properties"]["kind"].get("const") == "macos"
        )
        self.assertFalse(macos_schema["additionalProperties"])
        capabilities = macos_schema["properties"]["capabilities"]
        self.assertEqual(set(capabilities["required"]), macos.CAPABILITY_KEYS)
        self.assertEqual(set(capabilities["properties"]), macos.CAPABILITY_KEYS)


if __name__ == "__main__":
    unittest.main()
