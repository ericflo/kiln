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
            self.assertEqual(thermal._parse_host_temperature(policy, row), 72_050)
            row["HighPrecisionTemperature"] = 3600
            with self.assertRaisesRegex(
                thermal.ThermalGuardError,
                "Temperature=345 K, HighPrecisionTemperature=3600 tenths K",
            ):
                thermal._parse_host_temperature(policy, row)

    def test_persistent_counter_uses_one_timestamped_category_snapshot(self) -> None:
        self.assertIn("$snapshot = $category.ReadCategory()", thermal.HOST_COUNTER_SCRIPT)
        self.assertIn(
            "thermal snapshot counter timestamps disagree",
            thermal.HOST_COUNTER_SCRIPT,
        )
        self.assertNotIn(".NextValue()", thermal.HOST_COUNTER_SCRIPT)

    def test_persistent_counter_json_rejects_duplicate_keys(self) -> None:
        with self.assertRaisesRegex(
            thermal.ThermalGuardError, "duplicate key"
        ):
            thermal._strict_json_loads(
                '{"Schema":"first","Schema":"second"}',
                "fixture counter",
            )

    def test_gpu_uuid_and_name_are_both_bound(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            policy = thermal.load_policy(self.write_policy(Path(directory)))

            class Counter:
                def __init__(self, name: str = "Test GPU") -> None:
                    self.name = name
                    self.closed = False

                def receipt_identity(self) -> dict[str, object]:
                    return {
                        "device": {
                            "uuid": "GPU-test",
                            "name": self.name,
                        }
                    }

                def read_temperature_millicelsius(self) -> int:
                    return 63_000

                def close(self) -> None:
                    self.closed = True

            accepted = Counter()
            temperature = thermal.NvmlTemperatureCounter(
                policy,
                counter_factory=lambda *_args, **_kwargs: accepted,
            )
            self.assertEqual(temperature.read_millicelsius(), 63_000)
            temperature.close()
            self.assertTrue(accepted.closed)

            rejected = Counter("Different GPU")
            with self.assertRaisesRegex(
                thermal.ThermalGuardError, "identity"
            ):
                thermal.NvmlTemperatureCounter(
                    policy,
                    counter_factory=lambda *_args, **_kwargs: rejected,
                )
            self.assertTrue(rejected.closed)

    def test_persistent_host_counter_is_sequence_checked_and_reused(self) -> None:
        policy = thermal.validate_policy(policy_document())
        fixture = r"""
import json
import sys

print(json.dumps({
    "Schema": "kiln.wsl2-host-thermal-counter.v2",
    "CpuName": "Test CPU",
    "Name": "\\_TZ.THRM",
    "CounterNames": [
        "Temperature",
        "High Precision Temperature",
        "% Passive Limit",
        "Throttle Reasons",
    ],
}), flush=True)
for line in sys.stdin:
    sequence = int(line)
    print(json.dumps({
        "Schema": "kiln.wsl2-host-thermal-counter.v2",
        "Sequence": sequence,
        "Name": "\\_TZ.THRM",
        "Timestamp100nSec": 1000 + sequence,
        "Temperature": 345,
        "HighPrecisionTemperature": 3452,
        "PercentPassiveLimit": 100,
        "ThrottleReasons": 0,
    }), flush=True)
"""

        def process_factory(*_args, **kwargs):
            return thermal.subprocess.Popen(
                [sys.executable, "-c", fixture],
                stdin=kwargs["stdin"],
                stdout=kwargs["stdout"],
                stderr=kwargs["stderr"],
                bufsize=kwargs["bufsize"],
            )

        with thermal.WindowsThermalCounter(
            policy,
            process_factory=process_factory,
        ) as counter:
            self.assertEqual(counter.read_millicelsius(), 72_050)
            self.assertEqual(counter.read_millicelsius(), 72_050)

    def test_persistent_host_counter_rejects_stale_snapshot_timestamp(self) -> None:
        policy = thermal.validate_policy(policy_document())
        fixture = r"""
import json
import sys

print(json.dumps({
    "Schema": "kiln.wsl2-host-thermal-counter.v2",
    "CpuName": "Test CPU",
    "Name": "\\_TZ.THRM",
    "CounterNames": [
        "Temperature",
        "High Precision Temperature",
        "% Passive Limit",
        "Throttle Reasons",
    ],
}), flush=True)
for line in sys.stdin:
    sequence = int(line)
    print(json.dumps({
        "Schema": "kiln.wsl2-host-thermal-counter.v2",
        "Sequence": sequence,
        "Name": "\\_TZ.THRM",
        "Timestamp100nSec": 1000,
        "Temperature": 345,
        "HighPrecisionTemperature": 3452,
        "PercentPassiveLimit": 100,
        "ThrottleReasons": 0,
    }), flush=True)
"""

        def process_factory(*_args, **kwargs):
            return thermal.subprocess.Popen(
                [sys.executable, "-c", fixture],
                stdin=kwargs["stdin"],
                stdout=kwargs["stdout"],
                stderr=kwargs["stderr"],
                bufsize=kwargs["bufsize"],
            )

        with thermal.WindowsThermalCounter(
            policy,
            process_factory=process_factory,
        ) as counter:
            self.assertEqual(counter.read_millicelsius(), 72_050)
            with self.assertRaisesRegex(
                thermal.ThermalGuardError,
                "timestamp did not advance",
            ):
                counter.read_millicelsius()

    def test_persistent_host_counter_binds_registry_cpu_identity(self) -> None:
        policy = thermal.validate_policy(policy_document())
        fixture = r"""
import json

print(json.dumps({
    "Schema": "kiln.wsl2-host-thermal-counter.v2",
    "CpuName": "Different CPU",
    "Name": "\\_TZ.THRM",
    "CounterNames": [
        "Temperature",
        "High Precision Temperature",
        "% Passive Limit",
        "Throttle Reasons",
    ],
}), flush=True)
"""

        def process_factory(*_args, **kwargs):
            return thermal.subprocess.Popen(
                [sys.executable, "-c", fixture],
                stdin=kwargs["stdin"],
                stdout=kwargs["stdout"],
                stderr=kwargs["stderr"],
                bufsize=kwargs["bufsize"],
            )

        with self.assertRaisesRegex(
            thermal.ThermalGuardError, "exact identity"
        ):
            thermal.WindowsThermalCounter(
                policy,
                process_factory=process_factory,
            )

    def test_persistent_host_counter_rejects_wrong_sequence(self) -> None:
        policy = thermal.validate_policy(policy_document())
        fixture = r"""
import json
import sys

print(json.dumps({
    "Schema": "kiln.wsl2-host-thermal-counter.v2",
    "CpuName": "Test CPU",
    "Name": "\\_TZ.THRM",
    "CounterNames": [
        "Temperature",
        "High Precision Temperature",
        "% Passive Limit",
        "Throttle Reasons",
    ],
}), flush=True)
for _line in sys.stdin:
    print(json.dumps({
        "Schema": "kiln.wsl2-host-thermal-counter.v2",
        "Sequence": 99,
        "Name": "\\_TZ.THRM",
        "Timestamp100nSec": 1000,
        "Temperature": 345,
        "HighPrecisionTemperature": 3452,
        "PercentPassiveLimit": 100,
        "ThrottleReasons": 0,
    }), flush=True)
"""

        def process_factory(*_args, **kwargs):
            return thermal.subprocess.Popen(
                [sys.executable, "-c", fixture],
                stdin=kwargs["stdin"],
                stdout=kwargs["stdout"],
                stderr=kwargs["stderr"],
                bufsize=kwargs["bufsize"],
            )

        with thermal.WindowsThermalCounter(
            policy,
            process_factory=process_factory,
        ) as counter:
            with self.assertRaisesRegex(
                thermal.ThermalGuardError, "schema or sequence"
            ):
                counter.read_millicelsius()

    def test_persistent_host_counter_response_timeout_fails_closed(self) -> None:
        policy = thermal.validate_policy(policy_document())
        fixture = r"""
import json
import sys

print(json.dumps({
    "Schema": "kiln.wsl2-host-thermal-counter.v2",
    "CpuName": "Test CPU",
    "Name": "\\_TZ.THRM",
    "CounterNames": [
        "Temperature",
        "High Precision Temperature",
        "% Passive Limit",
        "Throttle Reasons",
    ],
}), flush=True)
for _line in sys.stdin:
    pass
"""

        def process_factory(*_args, **kwargs):
            return thermal.subprocess.Popen(
                [sys.executable, "-c", fixture],
                stdin=kwargs["stdin"],
                stdout=kwargs["stdout"],
                stderr=kwargs["stderr"],
                bufsize=kwargs["bufsize"],
            )

        counter = thermal.WindowsThermalCounter(
            policy,
            process_factory=process_factory,
        )
        try:
            with mock.patch.object(
                thermal,
                "HOST_COUNTER_TIMEOUT_SECONDS",
                0.01,
            ):
                with self.assertRaisesRegex(
                    thermal.ThermalGuardError, "timed out"
                ):
                    counter.read_millicelsius()
        finally:
            counter.close()

    def test_persistent_sampler_closes_host_when_nvml_initialization_fails(self) -> None:
        policy = thermal.validate_policy(policy_document())
        host = mock.Mock()
        with mock.patch.object(
            thermal,
            "WindowsThermalCounter",
            return_value=host,
        ), mock.patch.object(
            thermal,
            "NvmlTemperatureCounter",
            side_effect=thermal.ThermalGuardError("fixture NVML failure"),
        ):
            with self.assertRaisesRegex(
                thermal.ThermalGuardError, "fixture NVML failure"
            ):
                thermal.PersistentThermalSampler(policy)
        host.close.assert_called_once_with()

    def test_persistent_sampler_samples_and_closes_both_sources(self) -> None:
        policy = thermal.validate_policy(policy_document())
        host = mock.Mock()
        host.read_millicelsius.return_value = 72_050
        gpu = mock.Mock()
        gpu.read_millicelsius.return_value = 63_000
        with mock.patch.object(
            thermal,
            "WindowsThermalCounter",
            return_value=host,
        ), mock.patch.object(
            thermal,
            "NvmlTemperatureCounter",
            return_value=gpu,
        ):
            with thermal.PersistentThermalSampler(policy) as sampler:
                observed = sampler.sample()
                self.assertEqual(observed.host_millicelsius, 72_050)
                self.assertEqual(observed.gpu_millicelsius, 63_000)
        host.close.assert_called_once_with()
        gpu.close.assert_called_once_with()

    def test_limits_are_inclusive_and_cannot_be_disabled(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            policy = thermal.load_policy(self.write_policy(Path(directory)))
            at_limit = thermal.ThermalSample(0.0, 95_000, 60_000)
            with self.assertRaisesRegex(thermal.ThermalGuardError, "host temperature"):
                thermal._check_limits(at_limit, policy)
            at_gpu_limit = thermal.ThermalSample(0.0, 80_000, 85_000)
            with self.assertRaisesRegex(thermal.ThermalGuardError, "GPU temperature"):
                thermal._check_limits(at_gpu_limit, policy)

    def test_pacing_preflight_requires_the_stable_resume_boundary(self) -> None:
        document = policy_document(pacing=True)
        assert isinstance(document["pacing"], dict)
        document["pacing"]["resume_stable_samples"] = 3
        document["content_sha256"] = thermal._canonical_policy_hash(document)
        policy = thermal.validate_policy(document)
        observed = iter(
            [
                thermal.ThermalSample(0.0, 80_000, 60_000),
                thermal.ThermalSample(1.0, 71_000, 69_000),
                thermal.ThermalSample(2.0, 71_000, 69_000),
                thermal.ThermalSample(3.0, 71_000, 69_000),
            ]
        )
        with mock.patch.object(thermal.time, "sleep"):
            result = thermal._stable_preflight(policy, lambda: next(observed))
        self.assertEqual(result.sample_count, 4)
        self.assertEqual(result.ending.host_millicelsius, 71_000)
        self.assertEqual(result.peak_host_millicelsius, 80_000)
        self.assertEqual(result.peak_gpu_millicelsius, 69_000)

    def test_pacing_preflight_timeout_does_not_launch_a_child(self) -> None:
        policy = thermal.validate_policy(policy_document(pacing=True))
        current = thermal.ThermalSample(0.0, 80_000, 60_000)
        with mock.patch.object(
            thermal.time,
            "monotonic",
            side_effect=[0.0, 6.0],
        ), self.assertRaisesRegex(
            thermal.ThermalGuardError,
            "stable preflight thermal boundary timed out",
        ):
            thermal._stable_preflight(policy, lambda: current)

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
        sampler = mock.MagicMock()
        sampler.__enter__.return_value = sampler
        sampler.sample.side_effect = observed
        stderr = io.StringIO()
        with mock.patch.object(
            thermal,
            "verify_host_identity",
        ), mock.patch.object(
            thermal,
            "PersistentThermalSampler",
            return_value=sampler,
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
