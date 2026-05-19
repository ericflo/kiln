"""Rubric sanity check: known-good calibration transcripts should score
substantially higher than known-bad ones.

Reads calibration/good.jsonl and calibration/bad.jsonl. Each line is a
record:
  {
    "task": {...},                      # task spec (gold, target_bytes, ...)
    "transcript": [...],                # synthetic pi session events
  }

Exits 0 only when min(good_scores) > max(bad_scores) AND
mean(good) - mean(bad) > 0.30.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE))
import rubric  # noqa: E402


def score_set(path: Path) -> list[float]:
    if not path.exists():
        print(f"WARN: {path} missing — assuming empty set")
        return []
    scores: list[float] = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            task = rec["task"]
            transcript = rec["transcript"]
            score = rubric.score_rollout(transcript, "/tmp/n/a", task)
            scores.append(score["composite"])
            print(f"  {rec.get('name','?'):30s} composite={score['composite']:.3f} "
                  f"outcome={score['outcome']:.2f} eff={score['efficiency']:.2f} "
                  f"tc={score['tool_choice']:.2f} gr={score['grounding']:.2f}")
    return scores


def main():
    good_path = HERE / "calibration/good.jsonl"
    bad_path = HERE / "calibration/bad.jsonl"
    print("=== GOOD ===")
    good_scores = score_set(good_path)
    print("=== BAD ===")
    bad_scores = score_set(bad_path)

    if not good_scores or not bad_scores:
        print("SANITY: missing calibration set", file=sys.stderr)
        sys.exit(2)

    mean_good = sum(good_scores) / len(good_scores)
    mean_bad = sum(bad_scores) / len(bad_scores)
    sep = min(good_scores) - max(bad_scores)
    print()
    print(f"mean(good) = {mean_good:.3f}")
    print(f"mean(bad)  = {mean_bad:.3f}")
    print(f"min(good)  = {min(good_scores):.3f}")
    print(f"max(bad)   = {max(bad_scores):.3f}")
    print(f"separation = {sep:.3f}")
    ok = sep > 0 and (mean_good - mean_bad) > 0.30
    print("PASS" if ok else "FAIL")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
