"""Task-corpus builder — scaffold for pi-test-interpretation.

Round 2 contract: writes two files to datasets/:

  datasets/train.tasks.jsonl    (committed)
  datasets/eval.tasks.jsonl     (GITIGNORED — blind-eval firewall)

Each line is a JSON object whose schema is documented in capability.md
under `## Task shape`.

This file is a SCAFFOLD. Use ../pi-doctest/build_corpus.py (humaneval
derivation) or ../pi-faithful-completion/build_corpus.py (mixed sources)
as a reference and adapt to this cap.
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
        "pi-test-interpretation: build_corpus.py is a scaffold. Fill in the corpus generator "
        "per capability.md `## Task shape`. Seed deterministically so the eval "
        "split is reproducible."
    )


if __name__ == "__main__":
    main()
