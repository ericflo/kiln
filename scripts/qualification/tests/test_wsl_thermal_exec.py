from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock


QUALIFICATION_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(QUALIFICATION_DIR))
SPEC = importlib.util.spec_from_file_location(
    "qualification_wsl_thermal_exec",
    QUALIFICATION_DIR / "wsl_thermal_exec.py",
)
assert SPEC is not None and SPEC.loader is not None
thermal = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = thermal
SPEC.loader.exec_module(thermal)


def policy_document() -> dict[str, object]:
    value: dict[str, object] = {
        "schema": thermal.SCHEMA,
        "id": "test-wsl2-policy-v1",
        "content_sha256": "",
        "host": {
            "cpu_name": "Test CPU",
            "thermal_zone_name": "\\_TZ.THRM",
            "limit_millicelsius": 95_000,
            "vendor_tjunction_millicelsius": 110_000,
        },
        "gpu": {
            "name": "Test GPU",
            "uuid": "GPU-test",
            "limit_millicelsius": 85_000,
        },
        "poll_interval_ms": 100,
        "safe_handoff": {
            "host_target_millicelsius": 85_000,
            "gpu_target_millicelsius": 75_000,
            "stable_samples": 2,
            "timeout_seconds": 5,
        },
    }
    hashed = dict(value)
    hashed.pop("content_sha256")
    payload = json.dumps(
        hashed,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("ascii")
    value["content_sha256"] = f"sha256:{hashlib.sha256(payload).hexdigest()}"
    return value


class WslThermalExecTests(unittest.TestCase):
    def write_policy(self, root: Path) -> Path:
        path = root / "policy.json"
        path.write_text(json.dumps(policy_document()), encoding="ascii")
        return path

    def test_content_hash_and_exact_schema_fail_closed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = self.write_policy(root)
            parsed = thermal.load_policy(path)
            self.assertEqual(parsed.host_limit_millicelsius, 95_000)
            mutated = policy_document()
            mutated["poll_interval_ms"] = 101
            path.write_text(json.dumps(mutated), encoding="ascii")
            with self.assertRaisesRegex(thermal.ThermalGuardError, "hash mismatch"):
                thermal.load_policy(path)

            unknown = policy_document()
            unknown["disabled"] = True
            path.write_text(json.dumps(unknown), encoding="ascii")
            with self.assertRaisesRegex(thermal.ThermalGuardError, "unknown disabled"):
                thermal.load_policy(path)

    def test_windows_high_precision_temperature_is_tenths_kelvin(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            policy = thermal.load_policy(self.write_policy(Path(directory)))
            row = {
                "Name": "\\_TZ.THRM",
                "Temperature": 345,
                "HighPrecisionTemperature": 3452,
                "PercentPassiveLimit": 100,
                "ThrottleReasons": 0,
            }
            with mock.patch.object(thermal, "_powershell_json", return_value=row):
                self.assertEqual(thermal._host_temperature(policy), 72_050)
            row["HighPrecisionTemperature"] = 3600
            with mock.patch.object(thermal, "_powershell_json", return_value=row):
                with self.assertRaisesRegex(
                    thermal.ThermalGuardError, "disagree"
                ):
                    thermal._host_temperature(policy)

    def test_gpu_uuid_and_name_are_both_bound(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            policy = thermal.load_policy(self.write_policy(Path(directory)))
            with mock.patch.object(
                thermal,
                "_run",
                return_value="Test GPU, GPU-test, 63",
            ):
                self.assertEqual(thermal._gpu_temperature(policy), 63_000)
            with mock.patch.object(
                thermal,
                "_run",
                return_value="Different GPU, GPU-test, 63",
            ):
                with self.assertRaisesRegex(
                    thermal.ThermalGuardError, "identity mismatch"
                ):
                    thermal._gpu_temperature(policy)

    def test_limits_are_inclusive_and_cannot_be_disabled(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            policy = thermal.load_policy(self.write_policy(Path(directory)))
            at_limit = thermal.ThermalSample(0.0, 95_000, 60_000)
            with self.assertRaisesRegex(thermal.ThermalGuardError, "host temperature"):
                thermal._check_limits(at_limit, policy)
            at_gpu_limit = thermal.ThermalSample(0.0, 80_000, 85_000)
            with self.assertRaisesRegex(thermal.ThermalGuardError, "GPU temperature"):
                thermal._check_limits(at_gpu_limit, policy)


if __name__ == "__main__":
    unittest.main()
