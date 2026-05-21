"""Pi rollout driver — scaffold for pi-incremental-progress.

Round 2 pattern (kiln #10, #11): drive pi to produce a session JSONL,
call `kiln trajectory inspect` to validate, then score with rubric.py.

Reference implementations:
  - ../pi-doctest/rollout.py (single-turn)
  - ../pi-terminal-bench-lite/rollout.py (multi-turn)

This file is a SCAFFOLD. The next agent picking this cap up fills in
the task-specific pi driver. See ./capability.md `## Task shape` for
the input contract.
"""
from __future__ import annotations
import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "lib"))


def main():
    raise NotImplementedError(
        "pi-incremental-progress: rollout.py is a scaffold. See ./capability.md, "
        "../README.md (ECHO defaults), and ../pi-doctest/rollout.py (reference)."
    )


if __name__ == "__main__":
    main()
