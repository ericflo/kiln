"""Sample long real conversations from trajectories.db, materialize as
compaction tasks, and split into train / eval JSONLs.

Each line in the output is a single task ready for the rollout harness:

    {
      "task_id": "task_0000",
      "turn_id": "...",
      "session_id": "...",
      "model": "...",
      "input_tokens": int,
      "num_input_messages": int,
      "source_messages": [...],     # Anthropic-format messages
      "source_text": "...",         # pi-serialized form (cached for rubric)
      "ground_truth": {...},
    }

Length stratification (configurable): three buckets
    short:  30K-50K input tokens
    medium: 50K-80K input tokens
    long:   80K-120K input tokens

A "task" is one Anthropic-API turn's input *messages* (the trailing
"chosen response" from the production turn is discarded — we don't care
what the production model said; we care about the *context that needed
compaction*).
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sqlite3
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import task_scaffold


DEFAULT_DB = os.environ.get(
    "TRAJECTORIES_DB", "/data/apps/trajectory-trainer/trajectories.db"
)


STRATA = [
    ("short", 30_000, 50_000),
    ("medium", 50_000, 80_000),
    ("long", 80_000, 120_000),
]


def sample_turn_ids(conn: sqlite3.Connection, n_per_stratum: int, split: str, seed: int) -> list[str]:
    """Sample turn IDs from each stratum + split."""
    random.seed(seed)
    out: list[str] = []
    for name, lo, hi in STRATA:
        rows = conn.execute(
            """
            SELECT id FROM turns
            WHERE split = ?
              AND input_tokens BETWEEN ? AND ?
              AND num_tools >= 3
              AND num_input_messages >= 20
              AND model LIKE 'claude-%'
            ORDER BY RANDOM()
            LIMIT ?
            """,
            (split, lo, hi, n_per_stratum * 3),  # over-fetch, then sample
        ).fetchall()
        ids = [r[0] for r in rows]
        random.shuffle(ids)
        out.extend(ids[:n_per_stratum])
    return out


def fetch_turn(conn: sqlite3.Connection, turn_id: str) -> dict:
    row = conn.execute(
        """
        SELECT id, session_id, model, input_tokens, num_input_messages, num_tools, messages_json
        FROM turns
        WHERE id = ?
        """,
        (turn_id,),
    ).fetchone()
    if not row:
        return {}
    return {
        "turn_id": row[0],
        "session_id": row[1],
        "model": row[2],
        "input_tokens": row[3],
        "num_input_messages": row[4],
        "num_tools": row[5],
        "source_messages": json.loads(row[6]),
    }


def materialize_task(turn: dict, task_id: str) -> dict | None:
    """Convert one trajectories.db turn into a compaction task."""
    msgs = turn.get("source_messages") or []
    if not msgs:
        return None

    # Sanity: require multiple roles + tool calls
    roles = {m.get("role") for m in msgs}
    if "user" not in roles or "assistant" not in roles:
        return None

    serialized = task_scaffold.serialize_conversation(msgs)
    if len(serialized) < 4_000:  # too short to need compaction
        return None
    if len(serialized) > 250_000:  # huge — would blow the model's input
        return None

    gt = task_scaffold.extract_ground_truth(msgs)

    # Require non-trivial ground truth — at least one path or identifier
    if not gt["source_paths"] and not gt["source_identifiers"]:
        return None

    return {
        "task_id": task_id,
        "turn_id": turn["turn_id"],
        "session_id": turn["session_id"],
        "model": turn["model"],
        "input_tokens": turn["input_tokens"],
        "num_input_messages": turn["num_input_messages"],
        "num_tools": turn["num_tools"],
        "source_messages": msgs,
        "source_text": serialized,
        "ground_truth": gt,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default=DEFAULT_DB)
    ap.add_argument("--out-dir", default="datasets")
    ap.add_argument("--train-per-stratum", type=int, default=15)
    ap.add_argument("--eval-per-stratum", type=int, default=8)
    ap.add_argument("--seed", type=int, default=3141592653)
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    conn = sqlite3.connect(args.db)
    conn.row_factory = sqlite3.Row

    train_ids = sample_turn_ids(conn, args.train_per_stratum, split="train", seed=args.seed)
    eval_ids = sample_turn_ids(conn, args.eval_per_stratum, split="val", seed=args.seed + 1)
    # If val pool is empty for a stratum, fall back to test
    if not eval_ids:
        eval_ids = sample_turn_ids(conn, args.eval_per_stratum, split="test", seed=args.seed + 1)

    def write_set(ids: list[str], prefix: str, out_path: Path) -> int:
        n = 0
        skipped = 0
        with out_path.open("w") as f:
            for i, tid in enumerate(ids):
                turn = fetch_turn(conn, tid)
                if not turn:
                    skipped += 1
                    continue
                task = materialize_task(turn, f"task_{n:04d}")
                if task is None:
                    skipped += 1
                    continue
                f.write(json.dumps(task, ensure_ascii=False) + "\n")
                n += 1
        print(f"wrote {n} tasks to {out_path} (skipped {skipped})")
        return n

    train_out = out_dir / "train.tasks.jsonl"
    eval_out = out_dir / "eval.tasks.jsonl"
    n_train = write_set(train_ids, "train", train_out)
    n_eval = write_set(eval_ids, "eval", eval_out)

    print(f"summary: train={n_train}, eval={n_eval}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
