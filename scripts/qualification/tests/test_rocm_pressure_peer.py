from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path


QUALIFICATION_DIR = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "qualification_rocm_pressure_peer",
    QUALIFICATION_DIR / "rocm_pressure_peer.py",
)
assert SPEC is not None and SPEC.loader is not None
peer = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = peer
SPEC.loader.exec_module(peer)


class RocmPressurePeerTests(unittest.TestCase):
    def test_snapshot_uses_maximum_card_values_and_ignores_connectors(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for card, vram_total, vram_used, gtt_total, gtt_used in (
                ("card0", 96 * peer.GIB, 12 * peer.GIB, 15 * peer.GIB, 2 * peer.GIB),
                ("card1", 32 * peer.GIB, 4 * peer.GIB, 8 * peer.GIB, peer.GIB),
            ):
                device = root / card / "device"
                device.mkdir(parents=True)
                (device / "mem_info_vram_total").write_text(f"{vram_total}\n")
                (device / "mem_info_vram_used").write_text(f"{vram_used}\n")
                (device / "mem_info_gtt_total").write_text(f"{gtt_total}\n")
                (device / "mem_info_gtt_used").write_text(f"{gtt_used}\n")
            connector = root / "card0-DP-1" / "device"
            connector.mkdir(parents=True)
            (connector / "mem_info_vram_total").write_text(f"{999 * peer.GIB}\n")

            snapshot = peer.read_drm_memory_snapshot(root)

        self.assertEqual(snapshot.total_bytes, 111 * peer.GIB)
        self.assertEqual(snapshot.used_bytes, 14 * peer.GIB)
        self.assertEqual(snapshot.free_bytes, 97 * peer.GIB)
        self.assertAlmostEqual(snapshot.free_fraction, 97 / 111)

    def test_snapshot_rejects_missing_positive_total(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaisesRegex(peer.PressurePeerError, "no positive"):
                peer.read_drm_memory_snapshot(Path(tmp))

    def test_next_allocation_reaches_target_without_overshoot(self) -> None:
        snapshot = peer.DrmMemorySnapshot(
            total_bytes=100 * peer.GIB,
            used_bytes=91 * peer.GIB,
            free_bytes=9 * peer.GIB,
            free_fraction=0.09,
        )
        size = peer.next_allocation_bytes(
            snapshot,
            target_free_fraction=0.08,
            minimum_free_fraction=0.05,
            chunk_bytes=2 * peer.GIB,
            remaining_budget_bytes=4 * peer.GIB,
        )
        self.assertEqual(size, peer.GIB)

    def test_next_allocation_respects_chunk_budget_and_alignment(self) -> None:
        snapshot = peer.DrmMemorySnapshot(
            total_bytes=100 * peer.GIB,
            used_bytes=10 * peer.GIB,
            free_bytes=90 * peer.GIB,
            free_fraction=0.90,
        )
        size = peer.next_allocation_bytes(
            snapshot,
            target_free_fraction=0.08,
            minimum_free_fraction=0.05,
            chunk_bytes=513 * peer.MIB,
            remaining_budget_bytes=511 * peer.MIB + 1,
        )
        self.assertEqual(size, 510 * peer.MIB)

    def test_next_allocation_stops_at_or_below_target(self) -> None:
        snapshot = peer.DrmMemorySnapshot(
            total_bytes=100 * peer.GIB,
            used_bytes=92 * peer.GIB,
            free_bytes=8 * peer.GIB,
            free_fraction=0.08,
        )
        size = peer.next_allocation_bytes(
            snapshot,
            target_free_fraction=0.08,
            minimum_free_fraction=0.05,
            chunk_bytes=peer.GIB,
            remaining_budget_bytes=peer.GIB,
        )
        self.assertEqual(size, 0)

    def test_minimum_free_floor_fails_closed(self) -> None:
        safe = peer.DrmMemorySnapshot(1000, 949, 51, 0.051)
        peer.require_minimum_free(safe, 0.05)
        unsafe = peer.DrmMemorySnapshot(1000, 951, 49, 0.049)
        with self.assertRaisesRegex(peer.PressurePeerError, "safety floor"):
            peer.require_minimum_free(unsafe, 0.05)

    def test_ready_file_is_atomic_and_cannot_replace_evidence(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "ready.json"
            peer.write_ready(path, {"allocated_bytes": 123})
            self.assertEqual(json.loads(path.read_text()), {"allocated_bytes": 123})
            with self.assertRaises(FileExistsError):
                peer.write_ready(path, {"allocated_bytes": 456})
            self.assertEqual(json.loads(path.read_text()), {"allocated_bytes": 123})


if __name__ == "__main__":
    unittest.main()
