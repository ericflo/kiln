from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path


QUALIFICATION_DIR = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "qualification_cuda_pressure_peer",
    QUALIFICATION_DIR / "cuda_pressure_peer.py",
)
assert SPEC is not None and SPEC.loader is not None
peer = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = peer
SPEC.loader.exec_module(peer)


def valid_args() -> argparse.Namespace:
    return argparse.Namespace(
        ready_file=Path("ready.json"),
        release_file=Path("release.json"),
        device=0,
        target_free_mib=2048,
        minimum_free_mib=1536,
        chunk_mib=256,
        max_allocation_mib=8192,
        hold_seconds=300.0,
        poll_milliseconds=100,
        cuda_library=peer.CUDA_LIBRARY,
    )


class CudaPressurePeerTests(unittest.TestCase):
    def test_next_allocation_reaches_target_without_overshoot(self) -> None:
        snapshot = peer.CudaMemorySnapshot(
            total_bytes=16 * peer.GIB,
            free_bytes=3 * peer.GIB,
        )
        size = peer.next_allocation_bytes(
            snapshot,
            target_free_bytes=2 * peer.GIB,
            minimum_free_bytes=1536 * peer.MIB,
            chunk_bytes=2 * peer.GIB,
            remaining_budget_bytes=8 * peer.GIB,
        )
        self.assertEqual(size, peer.GIB)

    def test_next_allocation_respects_chunk_budget_and_alignment(self) -> None:
        snapshot = peer.CudaMemorySnapshot(
            total_bytes=16 * peer.GIB,
            free_bytes=10 * peer.GIB,
        )
        size = peer.next_allocation_bytes(
            snapshot,
            target_free_bytes=2 * peer.GIB,
            minimum_free_bytes=1536 * peer.MIB,
            chunk_bytes=513 * peer.MIB,
            remaining_budget_bytes=511 * peer.MIB + 1,
        )
        self.assertEqual(size, 510 * peer.MIB)

    def test_next_allocation_stops_at_target(self) -> None:
        snapshot = peer.CudaMemorySnapshot(
            total_bytes=16 * peer.GIB,
            free_bytes=2 * peer.GIB,
        )
        self.assertEqual(
            peer.next_allocation_bytes(
                snapshot,
                target_free_bytes=2 * peer.GIB,
                minimum_free_bytes=1536 * peer.MIB,
                chunk_bytes=peer.GIB,
                remaining_budget_bytes=peer.GIB,
            ),
            0,
        )

    def test_minimum_free_floor_fails_closed(self) -> None:
        peer.require_minimum_free(
            peer.CudaMemorySnapshot(16 * peer.GIB, 1536 * peer.MIB),
            1536 * peer.MIB,
        )
        with self.assertRaisesRegex(peer.PressurePeerError, "safety floor"):
            peer.require_minimum_free(
                peer.CudaMemorySnapshot(16 * peer.GIB, 1536 * peer.MIB - 1),
                1536 * peer.MIB,
            )

    def test_argument_contract_retains_target_floor_margin(self) -> None:
        peer.validate_args(valid_args())
        args = valid_args()
        args.target_free_mib = args.minimum_free_mib + 255
        with self.assertRaisesRegex(peer.PressurePeerError, "256 MiB"):
            peer.validate_args(args)
        args = valid_args()
        args.max_allocation_mib = 8193
        with self.assertRaisesRegex(peer.PressurePeerError, "8192"):
            peer.validate_args(args)

    def test_evidence_file_is_atomic_and_cannot_be_replaced(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "ready.json"
            peer.write_json_no_clobber(path, {"allocated_bytes": 123})
            self.assertEqual(
                json.loads(path.read_text()), {"allocated_bytes": 123}
            )
            with self.assertRaises(FileExistsError):
                peer.write_json_no_clobber(path, {"allocated_bytes": 456})
            self.assertEqual(
                json.loads(path.read_text()), {"allocated_bytes": 123}
            )


if __name__ == "__main__":
    unittest.main()
