#!/usr/bin/env python3
"""Prepare H5 pairwise final-format GRPO groups from H4 ideal traces.

This reads train-derived H4 SFT data only. Each group has two completions with
identical tool-action prefixes. The only supervised contrast is the final
assistant sentence, so the GRPO signal targets `format_compliance`.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent

BAD_FINALS = [
    "Done.",
    "Implemented.",
    "The requested change is complete.",
    "Finished.",
]


def split_final(text: str) -> tuple[str, str]:
    marker = "</think>"
    idx = text.rfind(marker)
    if idx < 0:
        raise ValueError("completion has no final </think> marker")
    end = idx + len(marker)
    prefix = text[:end]
    final = text[end:]
    if not final.strip():
        raise ValueError("completion has empty final text")
    return prefix, final


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default=str(ROOT / "datasets/sft.h4-ideal-traces.jsonl"))
    ap.add_argument("--out", default=str(ROOT / "datasets/grpo.h5-format-pairs.jsonl"))
    args = ap.parse_args()

    rows = []
    for idx, line in enumerate(Path(args.source).read_text().splitlines(), 1):
        if not line.strip():
            continue
        row = json.loads(line)
        messages = row["messages"]
        assistant = messages[-1]["content"]
        prefix, good_final = split_final(assistant)
        bad_final = BAD_FINALS[(idx - 1) % len(BAD_FINALS)]
        bad = prefix + bad_final
        rows.append(
            {
                "messages": messages[:-1],
                "completions": [
                    {
                        "text": assistant,
                        "reward": 1.0,
                        "_meta": {"kind": "format_positive"},
                    },
                    {
                        "text": bad,
                        "reward": 0.0,
                        "_meta": {
                            "kind": "format_negative",
                            "removed_final": good_final.strip(),
                        },
                    },
                ],
                "_meta": row.get("_meta", {}),
            }
        )

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w") as f:
        for row in rows:
            f.write(json.dumps(row, sort_keys=True) + "\n")

    lengths = [
        len(comp["text"])
        for row in rows
        for comp in row["completions"]
    ]
    print(f"wrote={out} groups={len(rows)} completions={len(lengths)}")
    if lengths:
        print(f"chars_min={min(lengths)} chars_max={max(lengths)}")


if __name__ == "__main__":
    main()
