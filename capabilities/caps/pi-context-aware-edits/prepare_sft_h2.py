#!/usr/bin/env python3
"""Prepare H2 short-positive SFT data from train rollouts.

This consumes training rollouts only. It never reads eval tasks or eval
transcripts. The dataset keeps one short high-reward completion per train
group and normalizes sandbox-specific absolute paths back to relative paths.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


SANDBOX_PREFIX_RE = re.compile(
    r"/tmp/pi-context-aware-edits-rollouts/[^/]+__g\d+/"
)


def normalize_completion(text: str) -> str:
    return SANDBOX_PREFIX_RE.sub("", text)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--source",
        default="/tmp/pi-context-aware-edits-iter-h1-local-echo-format-outcome-v2/rollouts/grpo-train.jsonl",
        help="AgenticGroup JSONL from train rollouts.",
    )
    ap.add_argument(
        "--out",
        default="datasets/sft.h2-positive-short.jsonl",
        help="Output SFT JSONL path.",
    )
    ap.add_argument("--min-reward", type=float, default=0.8)
    ap.add_argument("--max-chars", type=int, default=2900)
    args = ap.parse_args()

    source = Path(args.source)
    out = Path(args.out)
    rows = []

    for group_idx, line in enumerate(source.read_text().splitlines(), 1):
        if not line.strip():
            continue
        group = json.loads(line)
        best = None
        for completion_idx, completion in enumerate(group.get("completions") or [], 1):
            reward = float(completion.get("reward") or 0.0)
            if reward < args.min_reward:
                continue
            text = normalize_completion(str(completion.get("text") or ""))
            if len(text) > args.max_chars:
                continue
            candidate = (len(text), -reward, completion_idx, text)
            if best is None or candidate < best:
                best = candidate
        if best is None:
            continue

        _length, neg_reward, completion_idx, text = best
        messages = list(group.get("messages") or [])
        messages.append({"role": "assistant", "content": text})
        rows.append(
            {
                "messages": messages,
                "_meta": {
                    "source_group": group_idx,
                    "source_completion": completion_idx,
                    "reward": -neg_reward,
                    "completion_chars": len(text),
                },
            }
        )

    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w") as f:
        for row in rows:
            f.write(json.dumps(row, sort_keys=True) + "\n")

    rewards = [row["_meta"]["reward"] for row in rows]
    chars = [row["_meta"]["completion_chars"] for row in rows]
    print(f"wrote={out} examples={len(rows)}")
    if rows:
        print(f"reward_min={min(rewards):.4f} reward_max={max(rewards):.4f}")
        print(f"chars_min={min(chars)} chars_max={max(chars)}")


if __name__ == "__main__":
    main()
