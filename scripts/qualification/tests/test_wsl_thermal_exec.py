from __future__ import annotations

import hashlib
import importlib.util
import io
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


def policy_document(*, pacing: bool = False) -> dict[str, object]:
    value: dict[str, object] = {
        "schema": thermal.SCHEMA_V2 if pacing else thermal.SCHEMA_V1,
        "id": "test-wsl2-policy-v2" if pacing else "test-wsl2-policy-v1",
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
    if pacing:
        value["pacing"] = {
            "mode": "cgroup_freeze",
            "host_start_millicelsius": 80_000,
            "host_resume_millicelsius": 72_000,
            "gpu_start_millicelsius": 75_000,
            "gpu_resume_millicelsius": 70_000,
            "resume_stable_samples": 2,
            "timeout_seconds": 5,
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

    def test_in_memory_policy_validation_uses_the_same_closed_contract(self) -> None:
        parsed = thermal.validate_policy(policy_document())
        self.assertEqual(parsed.gpu_uuid, "GPU-test")

        unknown = policy_document()
        unknown["disabled"] = True
        with self.assertRaisesRegex(thermal.ThermalGuardError, "unknown disabled"):
            thermal.validate_policy(unknown)

    def test_v2_pacing_policy_is_closed_and_hysteretic(self) -> None:
        parsed = thermal.validate_policy(policy_document(pacing=True))
        self.assertEqual(parsed.schema, thermal.SCHEMA_V2)
        self.assertIsNotNone(parsed.pacing)
        assert parsed.pacing is not None
        self.assertEqual(parsed.pacing.host_start_millicelsius, 80_000)

        invalid = policy_document(pacing=True)
        assert isinstance(invalid["pacing"], dict)
        invalid["pacing"]["host_resume_millicelsius"] = 81_000
        invalid["content_sha256"] = thermal._canonical_policy_hash(invalid)
        with self.assertRaisesRegex(thermal.ThermalGuardError, "below start"):
            thermal.validate_policy(invalid)

        missing = policy_document(pacing=True)
        assert isinstance(missing["pacing"], dict)
        del missing["pacing"]["timeout_seconds"]
        missing["content_sha256"] = thermal._canonical_policy_hash(missing)
        with self.assertRaisesRegex(thermal.ThermalGuardError, "missing timeout_seconds"):
            thermal.validate_policy(missing)

    def test_v2_outer_supervisor_requires_matching_scope_pacing_binding(self) -> None:
        policy = thermal.validate_policy(policy_document(pacing=True))
        with self.assertRaisesRegex(
            thermal.ThermalGuardError,
            "trusted WSL2 scope supervisor",
        ):
            thermal._validate_pacing_scope_command(policy, ["/bin/true"])

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "policy.json"
            path.write_text(json.dumps(policy_document(pacing=True)), encoding="ascii")
            command = [
                sys.executable,
                str(QUALIFICATION_DIR / "wsl_scope_exec.py"),
                "--thermal-pacing-policy",
                str(path),
                "--",
                "/bin/true",
            ]
            thermal._validate_pacing_scope_command(policy, command)
            different = policy_document(pacing=True)
            different["id"] = "different-policy"
            different["content_sha256"] = thermal._canonical_policy_hash(different)
            path.write_text(json.dumps(different), encoding="ascii")
            with self.assertRaisesRegex(thermal.ThermalGuardError, "does not match"):
                thermal._validate_pacing_scope_command(policy, command)

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

    def test_trip_terminates_child_and_completes_safe_handoff(self) -> None:
        policy = thermal.validate_policy(policy_document())
        process = mock.Mock()
        process.poll.side_effect = [None, None, 3, 3]
        process.wait.return_value = 3
        observed = [
            thermal.ThermalSample(0.0, 80_000, 60_000),
            thermal.ThermalSample(0.1, 95_000, 61_000),
            thermal.ThermalSample(0.2, 95_000, 61_000),
            thermal.ThermalSample(0.3, 90_000, 61_000),
            thermal.ThermalSample(0.4, 84_000, 61_000),
            thermal.ThermalSample(0.5, 84_000, 61_000),
        ]
        stderr = io.StringIO()
        with mock.patch.object(
            thermal,
            "verify_host_identity",
        ), mock.patch.object(
            thermal,
            "sample",
            side_effect=observed,
        ), mock.patch.object(
            thermal.subprocess,
            "Popen",
            return_value=process,
        ), mock.patch.object(
            thermal.time,
            "sleep",
        ), mock.patch.object(
            thermal.signal,
            "signal",
            return_value=thermal.signal.SIG_DFL,
        ), mock.patch.object(
            thermal,
            "_terminate",
            wraps=thermal._terminate,
        ) as terminate, mock.patch(
            "sys.stderr",
            stderr,
        ):
            with self.assertRaisesRegex(
                thermal.ThermalGuardError,
                "host temperature",
            ):
                thermal.supervise(policy, ["fixture"])

        terminate.assert_called_once_with(process)
        events = [
            json.loads(line.removeprefix("wsl2-thermal: "))
            for line in stderr.getvalue().splitlines()
        ]
        self.assertEqual([item["event"] for item in events], ["preflight", "trip", "complete"])
        complete = events[-1]
        self.assertEqual(complete["supervision_outcome"], "thermal_trip")
        self.assertEqual(complete["sample_count"], 6)
        self.assertEqual(complete["peak_host_millicelsius"], 95_000)
        self.assertEqual(complete["ending_host_millicelsius"], 84_000)
        self.assertEqual(complete["safe_handoff_stable_samples"], 2)


if __name__ == "__main__":
    unittest.main()
