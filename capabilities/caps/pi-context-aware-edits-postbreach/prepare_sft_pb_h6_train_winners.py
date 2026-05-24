#!/usr/bin/env python3
"""Prepare PB-H6 SFT rows from train-only winning rollout trajectories."""
from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent
DEFAULT_SOURCE = Path(
    "/tmp/pi-context-aware-edits-postbreach-pb-h2-train-rollouts/grpo-train.jsonl"
)
DEFAULT_OUT = ROOT / "datasets/sft.pb-h6-train-winner-trajectories.jsonl"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return f"sha256:{digest.hexdigest()}"


def load_rows(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open() as f:
        for line_idx, line in enumerate(f, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            messages = row.get("messages")
            completions = row.get("completions")
            if not isinstance(messages, list) or not isinstance(completions, list):
                raise ValueError(f"line {line_idx}: expected messages and completions lists")
            rows.append(row)
    return rows


def build_sft_rows(
    rows: list[dict[str, Any]], source: Path, min_reward: float
) -> list[dict[str, Any]]:
    sft_rows: list[dict[str, Any]] = []
    source_sha256 = sha256_file(source)
    for group_idx, row in enumerate(rows):
        messages = row["messages"]
        completions = row["completions"]
        for completion_idx, completion in enumerate(completions):
            reward = float(completion.get("reward", 0.0))
            text = completion.get("text")
            if reward < min_reward:
                continue
            if not isinstance(text, str) or not text.strip():
                raise ValueError(
                    f"group {group_idx} completion {completion_idx}: empty winning text"
                )
            trajectory = completion.get("trajectory") or []
            sft_rows.append(
                {
                    "messages": [
                        *messages,
                        {
                            "role": "assistant",
                            "content": text,
                        },
                    ],
                    "_meta": {
                        "source": str(source),
                        "source_sha256": source_sha256,
                        "source_group_idx": group_idx,
                        "completion_idx": completion_idx,
                        "reward": reward,
                        "min_reward": min_reward,
                        "completion_chars": len(text),
                        "trajectory_events": (
                            len(trajectory) if isinstance(trajectory, list) else None
                        ),
                    },
                }
            )
    return sft_rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", default=str(DEFAULT_SOURCE))
    parser.add_argument("--out", default=str(DEFAULT_OUT))
    parser.add_argument("--min-reward", type=float, default=0.95)
    parser.add_argument("--min-examples", type=int, default=8)
    args = parser.parse_args()

    source = Path(args.source)
    out = Path(args.out)
    rows = load_rows(source)
    sft_rows = build_sft_rows(rows, source, args.min_reward)
    if len(sft_rows) < args.min_examples:
        raise RuntimeError(
            f"only {len(sft_rows)} examples met reward >= {args.min_reward}; "
            f"minimum is {args.min_examples}"
        )

    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w") as f:
        for row in sft_rows:
            f.write(json.dumps(row, ensure_ascii=True, sort_keys=True) + "\n")

    reward_counts = Counter(row["_meta"]["reward"] for row in sft_rows)
    group_counts = Counter(row["_meta"]["source_group_idx"] for row in sft_rows)
    lengths = [row["_meta"]["completion_chars"] for row in sft_rows]
    print(f"source={source}")
    print(f"source_sha256={sha256_file(source)}")
    print(f"out={out}")
    print(f"out_sha256={sha256_file(out)}")
    print(f"groups_read={len(rows)} winner_groups={len(group_counts)} examples={len(sft_rows)}")
    print(f"reward_counts={dict(sorted(reward_counts.items()))}")
    print(f"completion_chars_min={min(lengths)} completion_chars_max={max(lengths)}")


if __name__ == "__main__":
    main()
