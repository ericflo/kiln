"""rubric_sanity.py — calibration sanity check.

Reads:
  calibration/good.jsonl  (≥5 known-high-quality rollouts)
  calibration/bad.jsonl   (≥5 known-low-quality rollouts; one per §0 cheat)

Scores both with rubric.score_one. Asserts:
  - mean composite of good >= mean composite of bad + 0.20

If the margin is too small, the rubric is broken regardless of method.
run_stage.sh runs this before every iter and fails the iter on calibration failure.

Bypass with KILN_SKIP_RUBRIC_SANITY=1 only for early-development scaffolds.
"""

import json
import os
import sys
from pathlib import Path

import rubric


def _read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def main():
    if os.environ.get("KILN_SKIP_RUBRIC_SANITY") == "1":
        print("KILN_SKIP_RUBRIC_SANITY=1 set; skipping (use only for scaffolds)")
        return 0

    here = Path(__file__).parent
    good = _read_jsonl(here / "calibration" / "good.jsonl")
    bad = _read_jsonl(here / "calibration" / "bad.jsonl")

    if len(good) < 5 or len(bad) < 5:
        print(
            f"FAIL: need >=5 good (have {len(good)}) and >=5 bad (have {len(bad)}) "
            "fixtures in calibration/",
            file=sys.stderr,
        )
        return 1

    good_scores = [rubric.score_one(r).get("composite", 0.0) for r in good]
    bad_scores = [rubric.score_one(r).get("composite", 0.0) for r in bad]
    good_mean = sum(good_scores) / len(good_scores)
    bad_mean = sum(bad_scores) / len(bad_scores)
    margin = good_mean - bad_mean

    print(f"good mean: {good_mean:.4f} (n={len(good)})")
    print(f"bad  mean: {bad_mean:.4f} (n={len(bad)})")
    print(f"margin:    {margin:.4f}")

    if margin < 0.20:
        print(
            f"FAIL: margin {margin:.4f} < 0.20. Rubric does not separate "
            "good from bad cleanly. Tighten the rubric BEFORE training.",
            file=sys.stderr,
        )
        return 1

    # Per-sub-score breakdown
    if good_scores:
        good_subs = rubric.score_one(good[0])
        for sub in good_subs:
            if sub == "composite":
                continue
            gs = [rubric.score_one(r).get(sub, 0.0) for r in good]
            bs = [rubric.score_one(r).get(sub, 0.0) for r in bad]
            gm = sum(gs) / len(gs)
            bm = sum(bs) / len(bs)
            print(f"  {sub:<28} good={gm:.4f} bad={bm:.4f} margin={gm-bm:+.4f}")

    print("PASS: rubric_sanity")
    return 0


if __name__ == "__main__":
    sys.exit(main())
