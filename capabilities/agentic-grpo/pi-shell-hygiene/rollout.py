"""Pi rollout driver — agentic-GRPO scaffold for pi-shell-hygiene.

Round 2 round-trip:

    tasks JSONL --> pi sessions --> kiln-canonical trajectory JSONL
                --> rubric --> grpo-train.jsonl

The canonical pi -> trajectory normalization NOW lives in kiln itself
(KILN_IMPROVEMENT_ISSUES.md #11 — `kiln_train::pi_trajectory`). New rollout
scripts should:

  1. drive pi to produce a session JSONL,
  2. call `kiln trajectory inspect <session.jsonl> --json` to validate that
     the trajectory has trainable action and env tokens,
  3. score the rollout with rubric.score_one,
  4. emit a ScoredRollout-shape JSONL for the trainer.

This file is a SCAFFOLD. The reference complete implementation lives at
`../pi-doctest/rollout.py` (single-turn) or `../pi-terminal-bench-lite/rollout.py`
(multi-turn). Adapt to this cap's task shape; do not re-invent the pi schema.
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
        "pi-shell-hygiene: rollout.py is a scaffold. Use ../pi-doctest/rollout.py as a "
        "reference and adapt to this cap's task shape. See ../../LAYOUT.md "
        "and ../README.md for the canonical round-2 flow."
    )


if __name__ == "__main__":
    main()
