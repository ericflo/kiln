"""Rubric sanity gate — round-2 mandatory.

Scores every entry in `calibration/good.jsonl` and
`calibration/bad.jsonl` with `rubric.score_one()` (or
`rubric.score_rollout(...)` for legacy caps) and asserts:

  min(good_composite) > max(bad_composite) + MARGIN

If this fails, the rubric is too lax (round-1 bug class: rubric
saturated and could not distinguish good behavior from cheap cheats).

By default `run_iter.sh` runs this BEFORE training; a failure here
prevents pod spend on a broken rubric.

Bypass with `KILN_SKIP_RUBRIC_SANITY=1` for explicit early dev work.
"""
from __future__ import annotations
import json
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
MARGIN = float(os.environ.get("RUBRIC_SANITY_MARGIN", "0.2"))


def _load_jsonl(p: Path) -> list[dict]:
    if not p.exists():
        return []
    return [json.loads(line) for line in p.read_text().splitlines() if line.strip()]


def _score(rubric, rollout: dict) -> float:
    if hasattr(rubric, "score_one"):
        d = rubric.score_one(rollout)
    elif hasattr(rubric, "score_rollout"):
        d = rubric.score_rollout(
            rollout.get("transcript", []),
            rollout.get("workdir", ""),
            rollout.get("task", {}),
        )
    else:
        raise AttributeError(
            "rubric.py must expose score_one(rollout) or score_rollout(transcript, workdir, task)"
        )
    if "composite" in d:
        return float(d["composite"])
    if "mean_composite" in d:
        return float(d["mean_composite"])
    raise KeyError("rubric output missing `composite`")


def main() -> int:
    if os.environ.get("KILN_SKIP_RUBRIC_SANITY"):
        print("rubric_sanity: skipped (KILN_SKIP_RUBRIC_SANITY set)")
        return 0

    sys.path.insert(0, str(HERE))
    try:
        import rubric  # type: ignore
    except ImportError as e:
        print(f"rubric_sanity: ERROR — cannot import rubric: {e}")
        return 2

    good = _load_jsonl(HERE / "calibration" / "good.jsonl")
    bad = _load_jsonl(HERE / "calibration" / "bad.jsonl")

    if not good or not bad:
        print(
            f"rubric_sanity: WARNING — calibration not populated "
            f"(good={len(good)}, bad={len(bad)}). "
            "Round-2 mandates >=5 good + >=5 bad fixtures."
        )
        # Don't fail — but warn loudly. (Will fail when populated and unseparated.)
        return 0

    good_scores = []
    bad_scores = []
    for r in good:
        try:
            good_scores.append(_score(rubric, r))
        except NotImplementedError:
            print("rubric_sanity: rubric is a scaffold (NotImplementedError) — skipping")
            return 0
    for r in bad:
        bad_scores.append(_score(rubric, r))

    g_min, g_max = min(good_scores), max(good_scores)
    b_min, b_max = min(bad_scores), max(bad_scores)
    sep = g_min - b_max
    print(f"rubric_sanity: good_scores  min={g_min:.4f}  max={g_max:.4f}")
    print(f"rubric_sanity: bad_scores   min={b_min:.4f}  max={b_max:.4f}")
    print(f"rubric_sanity: separation   {sep:+.4f}  (margin required: {MARGIN:.4f})")

    if sep < MARGIN:
        print(
            "\nrubric_sanity: FAIL — rubric cannot distinguish good from bad. "
            "Fix the rubric, expand calibration fixtures, or tighten weights. "
            "See ./capability.md `## Adversarial design (§0)` for the cheats "
            "that should score 0."
        )
        return 1
    print(f"rubric_sanity: PASS — separation {sep:.4f} >= margin {MARGIN}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
