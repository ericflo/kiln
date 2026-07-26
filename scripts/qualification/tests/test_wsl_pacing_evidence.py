import json
import stat
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
SCRIPT_DIR = ROOT / "scripts/qualification"
sys.path.insert(0, str(SCRIPT_DIR))

import wsl_pacing_evidence as pacing


POLICY_SHA256 = "sha256:" + "a" * 64


def event(
    sequence: int,
    pause_index: int,
    transition: str,
    started: float,
    observed: float,
) -> dict:
    active = transition == "started"
    return {
        "active": active,
        "duration_seconds": 0.0 if active else observed - started,
        "gpu_millicelsius": 64000,
        "host_millicelsius": 74050,
        "observed_monotonic_seconds": observed,
        "pause_index": pause_index,
        "policy_sha256": POLICY_SHA256,
        "schema": pacing.PACING_EVENT_SCHEMA,
        "sequence": sequence,
        "started_monotonic_seconds": started,
        "transition": transition,
    }


class WslPacingEvidenceTests(unittest.TestCase):
    def write_stream(self, root: Path, records: list[dict]) -> Path:
        path = root / "thermal-pacing-events.jsonl"
        path.write_text(
            "".join(
                json.dumps(record, sort_keys=True, separators=(",", ":"))
                + "\n"
                for record in records
            ),
            encoding="ascii",
        )
        path.chmod(0o400)
        return path

    def test_reads_exact_completed_intervals_and_overlap(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = self.write_stream(
                Path(temporary),
                [
                    event(0, 1, "started", 10.0, 10.0),
                    event(1, 1, "completed", 10.0, 20.0),
                    event(2, 2, "started", 30.0, 30.0),
                    event(3, 2, "completed", 30.0, 35.0),
                ],
            )
            snapshot = pacing.read_pacing_snapshot(
                {pacing.PACING_EVENTS_PATH_ENV: str(path)},
                expected_policy_sha256=POLICY_SHA256,
            )
        self.assertEqual(len(snapshot.records), 4)
        self.assertEqual(len(snapshot.completed_pauses), 2)
        self.assertEqual(snapshot.overlap_seconds(15.0, 32.0), 7.0)

    def test_rejects_unpaired_active_pause(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = self.write_stream(
                Path(temporary),
                [event(0, 1, "started", 10.0, 10.0)],
            )
            with self.assertRaisesRegex(
                pacing.WslPacingEvidenceError,
                "active pause",
            ):
                pacing.read_pacing_snapshot(
                    {pacing.PACING_EVENTS_PATH_ENV: str(path)},
                    expected_policy_sha256=POLICY_SHA256,
                )

    def test_rejects_policy_or_sequence_drift(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            records = [
                event(1, 1, "started", 10.0, 10.0),
                event(2, 1, "completed", 10.0, 20.0),
            ]
            path = self.write_stream(Path(temporary), records)
            with self.assertRaisesRegex(
                pacing.WslPacingEvidenceError,
                "schema, policy, or sequence",
            ):
                pacing.read_pacing_snapshot(
                    {pacing.PACING_EVENTS_PATH_ENV: str(path)},
                    expected_policy_sha256=POLICY_SHA256,
                )

    def test_rejects_duplicate_fields_and_partial_records(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "thermal-pacing-events.jsonl"
            path.write_text(
                '{"sequence":0,"sequence":0}\n',
                encoding="ascii",
            )
            path.chmod(0o400)
            with self.assertRaisesRegex(
                pacing.WslPacingEvidenceError,
                "malformed JSON",
            ):
                pacing.read_pacing_snapshot(
                    {pacing.PACING_EVENTS_PATH_ENV: str(path)},
                    expected_policy_sha256=POLICY_SHA256,
                )
            path.chmod(0o600)
            path.write_text('{"sequence":0}', encoding="ascii")
            path.chmod(0o400)
            with self.assertRaisesRegex(
                pacing.WslPacingEvidenceError,
                "partial record",
            ):
                pacing.read_pacing_snapshot(
                    {pacing.PACING_EVENTS_PATH_ENV: str(path)},
                    expected_policy_sha256=POLICY_SHA256,
                )

    def test_rejects_writable_stream(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = self.write_stream(Path(temporary), [])
            path.chmod(stat.S_IRUSR | stat.S_IWUSR)
            with self.assertRaisesRegex(
                pacing.WslPacingEvidenceError,
                "unsafe type, ownership, or mode",
            ):
                pacing.read_pacing_snapshot(
                    {pacing.PACING_EVENTS_PATH_ENV: str(path)},
                    expected_policy_sha256=POLICY_SHA256,
                )

    def test_rejects_missing_or_relative_path(self) -> None:
        with self.assertRaisesRegex(
            pacing.WslPacingEvidenceError,
            "unavailable",
        ):
            pacing.read_pacing_snapshot(
                {},
                expected_policy_sha256=POLICY_SHA256,
            )
        with self.assertRaisesRegex(
            pacing.WslPacingEvidenceError,
            "normalized and absolute",
        ):
            pacing.read_pacing_snapshot(
                {pacing.PACING_EVENTS_PATH_ENV: "events.jsonl"},
                expected_policy_sha256=POLICY_SHA256,
            )


if __name__ == "__main__":
    unittest.main()
