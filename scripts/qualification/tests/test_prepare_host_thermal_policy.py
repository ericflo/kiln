from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock


ROOT = Path(__file__).resolve().parents[3]
QUALIFICATION_DIR = ROOT / "scripts" / "qualification"
if str(QUALIFICATION_DIR) not in sys.path:
    sys.path.insert(0, str(QUALIFICATION_DIR))
SCRIPT = QUALIFICATION_DIR / "prepare_host_thermal_policy.py"
SPEC = importlib.util.spec_from_file_location("prepare_host_thermal_policy", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
prepare = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(prepare)


def add_sensor(
    root: Path,
    index: int,
    *,
    name: str = "coretemp",
    label: str = "Package id 0",
    temperature: int = 45_000,
) -> Path:
    sensor = root / f"hwmon{index}"
    sensor.mkdir(parents=True)
    (sensor / "name").write_text(name + "\n", encoding="utf-8")
    (sensor / "temp1_label").write_text(label + "\n", encoding="utf-8")
    input_path = sensor / "temp1_input"
    input_path.write_text(f"{temperature}\n", encoding="utf-8")
    return input_path


def create_args(root: Path, output: Path) -> argparse.Namespace:
    return argparse.Namespace(
        hwmon_root=root,
        id="rtx4090-laptop-serving-hard-limit-v1",
        hwmon_name="coretemp",
        label="Package id 0",
        limit_millicelsius=90_000,
        poll_interval_ms=250,
        safe_handoff_target_millicelsius=60_000,
        safe_handoff_stable_samples=8,
        safe_handoff_timeout_seconds=300.0,
        phase_settlement_timeout_seconds=300.0,
        output=output,
    )


class HostThermalPolicyPreparationTests(unittest.TestCase):
    def test_inventory_is_stable_and_marks_ambiguous_selectors(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            add_sensor(root, 7, temperature=47_000)
            add_sensor(root, 2, temperature=43_000)
            result = prepare.inventory(root)
        self.assertEqual(result["schema"], prepare.INVENTORY_SCHEMA)
        self.assertEqual(result["sensor_count"], 2)
        self.assertEqual(
            [item["temperature_millicelsius"] for item in result["sensors"]],
            [43_000, 47_000],
        )
        self.assertTrue(
            all(not item["selector_resolves_uniquely"] for item in result["sensors"])
        )

    def test_inventory_rejects_a_label_without_an_input(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            sensor = root / "hwmon0"
            sensor.mkdir()
            (sensor / "name").write_text("coretemp\n", encoding="utf-8")
            (sensor / "temp1_label").write_text("Package id 0\n", encoding="utf-8")
            with self.assertRaisesRegex(prepare.PreparationError, "has no input"):
                prepare.inventory(root)

    def test_policy_is_content_hashed_resolved_and_published_without_clobber(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            input_path = add_sensor(root, 4)
            output = root / "policy.json"
            args = create_args(root, output)
            normalized, resolved, temperature = prepare.build_policy(args)
            prepare.publish_no_clobber(output, normalized)
            self.assertEqual(resolved, input_path)
            self.assertEqual(temperature, 45_000)
            self.assertRegex(normalized["content_sha256"], r"^sha256:[0-9a-f]{64}$")
            self.assertEqual(
                json.loads(output.read_text(encoding="ascii")),
                normalized,
            )
            self.assertEqual(output.stat().st_mode & 0o777, 0o644)
            with mock.patch.object(prepare.policy, "wait_for_prelaunch_cooldown"):
                self.assertEqual(
                    prepare.cargo_fields(output, root),
                    ("coretemp", "Package id 0", 90_000, 250),
                )
            with self.assertRaisesRegex(prepare.PreparationError, "refusing to replace"):
                prepare.publish_no_clobber(output, normalized)

    def test_policy_rejects_ambiguous_or_hot_sensor(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            add_sensor(root, 0)
            add_sensor(root, 1)
            args = create_args(root, root / "policy.json")
            with self.assertRaisesRegex(prepare.PreparationError, "resolved to 2"):
                prepare.build_policy(args)

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            add_sensor(root, 0, temperature=90_000)
            args = create_args(root, root / "policy.json")
            with self.assertRaisesRegex(prepare.PreparationError, "refusing policy"):
                prepare.build_policy(args)

    def test_policy_rejects_process_stop_pacing_by_construction(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            add_sensor(root, 0)
            normalized, _resolved, _temperature = prepare.build_policy(
                create_args(root, root / "policy.json")
            )
        self.assertEqual(normalized["pacing"], {"mode": "hard_limit_only"})

    def test_policy_materializer_bounds_limit_and_poll_cadence(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            add_sensor(root, 0)
            args = create_args(root, root / "policy.json")
            args.limit_millicelsius = 200_001
            with self.assertRaisesRegex(prepare.PreparationError, "1..=200000"):
                prepare.build_policy(args)
            args.limit_millicelsius = 90_000
            args.poll_interval_ms = 49
            with self.assertRaisesRegex(prepare.PreparationError, "50..=60000"):
                prepare.build_policy(args)

    def test_cargo_fields_requires_a_recorded_content_hash(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            add_sensor(root, 0)
            output = root / "policy.json"
            normalized, _resolved, _temperature = prepare.build_policy(
                create_args(root, output)
            )
            normalized.pop("content_sha256")
            output.write_text(json.dumps(normalized), encoding="utf-8")
            with self.assertRaisesRegex(prepare.PreparationError, "content-hashed"):
                prepare.cargo_fields(output, root)


if __name__ == "__main__":
    unittest.main()
