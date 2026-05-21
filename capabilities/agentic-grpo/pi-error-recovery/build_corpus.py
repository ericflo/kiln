"""Task-corpus builder — scaffold for pi-error-recovery.

Writes:
  datasets/train.tasks.jsonl    (committed)
  datasets/eval.tasks.jsonl     (GITIGNORED — blind-eval firewall)

See ./capability.md `## Task shape` for the per-task JSON schema.
"""
from __future__ import annotations
import json
import os
import random
from pathlib import Path

HERE = Path(__file__).resolve().parent
DATASETS = HERE / "datasets"


def main():
    DATASETS.mkdir(exist_ok=True)
    raise NotImplementedError(
        "pi-error-recovery: build_corpus.py is a scaffold. Generate train + eval splits "
        "per ./capability.md and seed deterministically."
    )


if __name__ == "__main__":
    main()
