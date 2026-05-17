#!/usr/bin/env python3
"""rubric_sanity.py — Phase 0 rubric calibration check.

Why this exists: rubrics fail silently. A regex that's too narrow, a
sub-score that paraphrase-penalizes the right answer, an extraction
heuristic that misses common forms — these don't surface until you've
already trained an adapter and watched it score zero on a sub-score
where the response is obviously fine.

This helper makes the agent commit, BEFORE training, to a list of
"obvious" cases the rubric must score correctly:

  good.jsonl  — manually-crafted responses the rubric should rate ≥0.7
  bad.jsonl   — manually-crafted responses the rubric should rate ≤0.3

If any case scores outside its band, the rubric is broken — fix it
before generating prompts or running baseline.

Usage:
  cd opd-cap.<slug>/
  python3 $SKILL/templates/rubric_sanity.py \
    --good calibration/good.jsonl \
    --bad calibration/bad.jsonl

Each calibration JSONL file has one object per line with the same
input format the eval uses (e.g. {"transcript": "...", "response": "..."}
for compaction; whatever your oracle expects). The script imports
`rubric.score_response` from the cwd and runs each row through it.

Exits with code 2 if any case fails its band. The agent must NOT
proceed to training until exit is 0.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path


def load_rubric():
    rubric_path = Path("rubric.py")
    if not rubric_path.exists():
        print("ERROR: rubric.py not found in cwd", file=sys.stderr)
        sys.exit(2)
    spec = importlib.util.spec_from_file_location("rubric", rubric_path)
    mod = importlib.util.module_from_spec(spec)
    sys.path.insert(0, ".")
    spec.loader.exec_module(mod)
    if not hasattr(mod, "score_response"):
        print("ERROR: rubric.py must export score_response(...)", file=sys.stderr)
        sys.exit(2)
    return mod


def check_band(rubric, path: Path, band: str, threshold: float) -> list[str]:
    """Return list of failure messages."""
    if not path.exists():
        return [f"missing file: {path}"]
    failures = []
    with path.open() as f:
        for i, line in enumerate(f, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            # Convention: row has all fields score_response needs except
            # the response keyword is "response". Other fields pass as kwargs.
            kwargs = dict(row)
            scores = rubric.score_response(**kwargs)
            comp = scores.get("composite")
            if band == "good" and comp < threshold:
                failures.append(
                    f"{path}:{i} GOOD case scored {comp:.3f} (need ≥{threshold}): "
                    f"sub-scores={ {k:round(v,3) for k,v in scores.items() if k!='composite'} }"
                )
            elif band == "bad" and comp > threshold:
                failures.append(
                    f"{path}:{i} BAD case scored {comp:.3f} (need ≤{threshold}): "
                    f"sub-scores={ {k:round(v,3) for k,v in scores.items() if k!='composite'} }"
                )
    return failures


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--good", default="calibration/good.jsonl",
                    help="Manually-crafted responses the rubric should score ≥0.7")
    ap.add_argument("--bad", default="calibration/bad.jsonl",
                    help="Manually-crafted responses the rubric should score ≤0.3")
    ap.add_argument("--good-floor", type=float, default=0.7)
    ap.add_argument("--bad-ceiling", type=float, default=0.3)
    args = ap.parse_args()

    rubric = load_rubric()
    failures = []
    failures += check_band(rubric, Path(args.good), "good", args.good_floor)
    failures += check_band(rubric, Path(args.bad), "bad", args.bad_ceiling)

    if failures:
        print("RUBRIC SANITY FAILED:")
        for f in failures:
            print(f"  - {f}")
        print(
            "\nThe rubric does not correctly classify obvious cases. "
            "Fix it before training — likely your patterns are too narrow "
            "or a sub-score uses the wrong distance metric."
        )
        sys.exit(2)
    print("rubric sanity OK: all good cases ≥ good-floor, all bad cases ≤ bad-ceiling")


if __name__ == "__main__":
    main()
