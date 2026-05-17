"""Filter the teacher fixture + prompts to sequences below a length threshold.

Used to subset down to OPD-friendly sizes (the current opd_train has no
gradient checkpointing, so long sequences OOM on a 48GB GPU). Run this once
to produce `train.opd.short.jsonl` + `teacher.fixture.short.jsonl`.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).parent


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in-fixture", default=str(ROOT / "datasets" / "teacher.fixture.jsonl"))
    ap.add_argument("--in-prompts", default=str(ROOT / "datasets" / "train.opd.jsonl"))
    ap.add_argument("--out-fixture", default=str(ROOT / "datasets" / "teacher.fixture.short.jsonl"))
    ap.add_argument("--out-prompts", default=str(ROOT / "datasets" / "train.opd.short.jsonl"))
    ap.add_argument("--max-tokens", type=int, default=400)
    args = ap.parse_args()

    fix = [json.loads(l) for l in open(args.in_fixture)]
    prm = [json.loads(l) for l in open(args.in_prompts)]
    assert len(fix) == len(prm), (len(fix), len(prm))

    kept_fix, kept_prm = [], []
    for f, p in zip(fix, prm):
        seq_len = len(f["tokens"])
        if seq_len <= args.max_tokens:
            kept_fix.append(f)
            kept_prm.append(p)
    print(f"input: {len(fix)} rows; kept {len(kept_fix)} (seqlen ≤ {args.max_tokens})")

    with open(args.out_fixture, "w") as fo:
        for f in kept_fix:
            fo.write(json.dumps(f) + "\n")
    with open(args.out_prompts, "w") as fo:
        for p in kept_prm:
            fo.write(json.dumps(p) + "\n")
    print(f"wrote {args.out_fixture} and {args.out_prompts}")


if __name__ == "__main__":
    main()
