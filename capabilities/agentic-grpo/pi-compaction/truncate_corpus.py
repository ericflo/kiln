"""Produce a truncated copy of the train/eval JSONLs.

Iter 1 found that kiln-train on 50K-token serialized conversations is
prohibitively slow on H100 (>10 min and < 10% progress on 9 groups).
Truncating the source to the last 30K chars (~7500 tokens) makes training
tractable while preserving the recent-tail context that pi compaction
most needs.

Usage:
    python3 truncate_corpus.py [--chars 30000]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def truncate(text: str, max_chars: int) -> str:
    if len(text) <= max_chars:
        return text
    return "[... earlier turns truncated ...]\n\n" + text[-max_chars:]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--chars", type=int, default=30_000,
                    help="Keep this many trailing characters of source_text.")
    args = ap.parse_args()
    for stem in ("train.tasks.jsonl", "eval.tasks.jsonl", "train-short.tasks.jsonl"):
        src = ROOT / "datasets" / stem
        if not src.exists():
            continue
        out_stem = stem.replace(".tasks.jsonl", "-trunc.tasks.jsonl")
        out = ROOT / "datasets" / out_stem
        n = 0
        with src.open() as f_in, out.open("w") as f_out:
            for line in f_in:
                t = json.loads(line)
                t["source_text"] = truncate(t["source_text"], args.chars)
                f_out.write(json.dumps(t, ensure_ascii=False) + "\n")
                n += 1
        print(f"wrote {n} tasks -> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
