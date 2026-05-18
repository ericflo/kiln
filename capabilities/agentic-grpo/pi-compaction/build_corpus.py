"""Sample long real conversations from trajectories.db, materialize as
compaction tasks, and split into train / eval JSONLs.

**Security note.** The trajectories.db carries real production
tool-result text. Tool results occasionally include leaked secrets
(RunPod API keys, GitHub PATs, OAuth tokens, etc.) that the agent
echoed into a shell. This script scrubs known secret prefixes from
`source_text` and `source_messages.tool_result` blocks before writing
the JSONL. The resulting `datasets/*.tasks.jsonl` is kept under
`.gitignore` so it never lands on GitHub — even with the scrubber,
treat it as sensitive and rebuild locally per session.

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
import re
import sqlite3
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import task_scaffold


# Known secret-ish patterns we mask before persisting the corpus.
SECRET_PATTERNS = [
    # RunPod API key
    (re.compile(r"rpa_[A-Za-z0-9]{30,}"), "rpa_<REDACTED>"),
    # GitHub Personal Access Token (classic + fine-grained)
    (re.compile(r"ghp_[A-Za-z0-9]{30,}"), "ghp_<REDACTED>"),
    (re.compile(r"github_pat_[A-Za-z0-9_]{30,}"), "github_pat_<REDACTED>"),
    (re.compile(r"gho_[A-Za-z0-9]{30,}"), "gho_<REDACTED>"),
    (re.compile(r"ghs_[A-Za-z0-9]{30,}"), "ghs_<REDACTED>"),
    (re.compile(r"ghu_[A-Za-z0-9]{30,}"), "ghu_<REDACTED>"),
    # OpenAI keys
    (re.compile(r"sk-(?:proj-)?[A-Za-z0-9_-]{30,}"), "sk-<REDACTED>"),
    # Anthropic keys
    (re.compile(r"sk-ant-[A-Za-z0-9_-]{30,}"), "sk-ant-<REDACTED>"),
    # Generic Bearer tokens that look key-like
    (re.compile(r"Bearer\s+[A-Za-z0-9_\-.]{30,}"), "Bearer <REDACTED>"),
    # AWS access key + secret (heuristic)
    (re.compile(r"AKIA[0-9A-Z]{16}"), "AKIA<REDACTED>"),
    # Generic 40+ char hex digest assigned to a TOKEN/KEY/SECRET var
    (re.compile(r"(?i)(token|secret|key|password)\s*[:=]\s*['\"]?[A-Fa-f0-9]{32,}['\"]?"),
     r"\1=<REDACTED>"),
]


def scrub_secrets(text: str) -> str:
    if not text:
        return text
    for pat, repl in SECRET_PATTERNS:
        text = pat.sub(repl, text)
    return text


def scrub_messages(messages: list[dict]) -> list[dict]:
    """Scrub secret patterns out of tool-result content (the main leak surface)."""
    out: list[dict] = []
    for m in messages:
        content = m.get("content")
        if isinstance(content, list):
            new_content = []
            for b in content:
                if isinstance(b, dict):
                    bc = b.get("content")
                    if isinstance(bc, str):
                        b = {**b, "content": scrub_secrets(bc)}
                    elif isinstance(bc, list):
                        b = {**b, "content": [
                            {**sub, "text": scrub_secrets(sub.get("text", ""))}
                            if isinstance(sub, dict) and "text" in sub else sub
                            for sub in bc
                        ]}
                    if b.get("type") == "text" and "text" in b:
                        b = {**b, "text": scrub_secrets(b["text"])}
                new_content.append(b)
            out.append({**m, "content": new_content})
        elif isinstance(content, str):
            out.append({**m, "content": scrub_secrets(content)})
        else:
            out.append(m)
    return out


DEFAULT_DB = os.environ.get(
    "TRAJECTORIES_DB", "/data/apps/trajectory-trainer/trajectories.db"
)


STRATA = [
    ("short", 30_000, 50_000),
    ("medium", 50_000, 80_000),
    ("long", 80_000, 120_000),
]


def sample_turn_ids(conn: sqlite3.Connection, n_per_stratum: int, split: str, seed: int) -> list[str]:
    """Sample turn IDs from each stratum + split.

    Uses idx_turns_split + idx_turns_token_stats (avoid ORDER BY RANDOM on
    the 38 GB table — full table scan would take ages). We use a LIMIT
    over-fetch then python-side shuffle for randomness.
    """
    random.seed(seed)
    out: list[str] = []
    for name, lo, hi in STRATA:
        # Two-step: first collect IDs at the index level (fast), then shuffle.
        rows = conn.execute(
            """
            SELECT id FROM turns
            WHERE split = ?
              AND input_tokens BETWEEN ? AND ?
              AND num_tools >= 3
              AND num_input_messages >= 20
              AND model LIKE 'claude-%'
            LIMIT ?
            """,
            (split, lo, hi, n_per_stratum * 20),
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

    # Scrub secrets BEFORE serialization so the source_text is clean too.
    msgs = scrub_messages(msgs)

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
